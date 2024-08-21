import logging
from typing import Dict, List, Optional
import torch
from torch.utils.data.sampler import Sampler

from psd2.utils import comm
import numpy as np
from collections import Counter
import itertools
import functools
import math
import copy
from scipy.optimize import linear_sum_assignment
import assign_cost_cuda

logger = logging.getLogger(__name__)


class APKSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        data_dicts: List[Dict],
        p: int,
        k: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self._size = len(data_dicts)
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self.bs = batch_size
        self.p = p
        self.k = k
        self.nb = math.ceil(self._size / self.bs)
        self.org_imgs_pids = []
        max_p = 0
        for dd in data_dicts:
            pids = np.array([ann["person_id"] for ann in dd["annotations"]])
            pids = pids[pids > -1].tolist()
            self.org_imgs_pids.append(pids)
            if len(pids) > max_p:
                max_p = len(pids)
        # padding
        for img_pids in self.org_imgs_pids:
            while len(img_pids) < max_p:
                img_pids.append(-2)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        # sequence should match the default BatchSampler
        g = torch.Generator()
        g.manual_seed(self._seed)

        while True:
            all_imgs_idxs = list(range(self._size))  # indices to data dict
            len_diff = self.bs * self.nb - len(all_imgs_idxs)
            if len_diff > 0:
                append_idxs = torch.randint(
                    0, self._size, (len_diff,), generator=g
                ).tolist()
                all_imgs_idxs.extend(append_idxs)
            all_imgs_idxs = np.array(all_imgs_idxs)
            if self._shuffle:
                seq = torch.randperm(len(all_imgs_idxs), generator=g).tolist()
                all_imgs_idxs = all_imgs_idxs[seq]
            batch_idxs_built = all_imgs_idxs[: self.nb]
            batch_data_built = [
                [copy.deepcopy(self.org_imgs_pids[idx])] for idx in batch_idxs_built
            ]  # init batch
            batch_idxs_built = np.expand_dims(
                np.array(batch_idxs_built), axis=-1
            ).tolist()
            data_left_idxs = all_imgs_idxs[self.nb :].tolist()
            data_left = copy.deepcopy(
                [self.org_imgs_pids[idx] for idx in data_left_idxs]
            )
            for _ in range(self.bs - 1):
                cost_m = self._assign_cost(data_left, batch_data_built)
                indices = linear_sum_assignment(cost_m)
                assigned_data_indices = []
                for i, j in zip(*indices):
                    assigned_data_indices.append(i)
                    batch_data_built[j].append(data_left[i])
                    batch_idxs_built[j].append(data_left_idxs[i])
                data_left_new = []
                data_left_idxs_new = []
                for di in range(len(data_left)):
                    if di not in assigned_data_indices:
                        data_left_new.append(data_left[di])
                        data_left_idxs_new.append(data_left_idxs[di])
                data_left = data_left_new
                data_left_idxs = data_left_idxs_new
            batch_idxs_seq = itertools.chain(*batch_idxs_built)
            yield from batch_idxs_seq

    def _assign_cost(self, data_left, batch_built):
        data_left_arr = np.array(data_left)  # l x p
        batch_built_arr = np.array(batch_built)  # t x b x p
        # may reduce memory cost
        fl_batch_built_arr = np.resize(
            batch_built_arr,
            (
                batch_built_arr.shape[0],
                batch_built_arr.shape[1] * batch_built_arr.shape[2],
            ),
        )  # t x bp
        """fl_batch_built_arr = np.expand_dims(fl_batch_built_arr, axis=1)  # t x 1 x bp
        ac_splits = []
        for i in range(fl_batch_built_arr.shape[0]):
            s_batch_built_arr = fl_batch_built_arr[i].copy()
            s_ac = assign_cost_cuda.assign_cost(
                data_left_arr, s_batch_built_arr, self.p, self.k
            )
            ac_splits.append(s_ac)  # l x 1
        ac = np.concatenate(ac_splits, axis=-1)"""
        ac = assign_cost_cuda.assign_cost(
            data_left_arr, fl_batch_built_arr, self.p, self.k
        )
        return ac


def pre_assign(
    sorted_dataset,
    batch_size,
    p,
    k,
    shuffle,
    save_path,
    drop_last=False,
    times=1024,
    seed: Optional[int] = None,
):
    if seed is None:
        seed = comm.shared_random_seed()
    seed = int(seed)
    size = len(sorted_dataset)
    if drop_last:
        nb = size // batch_size
    else:
        nb = math.ceil(size / batch_size)
    if comm.get_local_rank() != 0:
        comm.synchronize()
    else:
        org_imgs_pids = []
        max_p = 0
        for dd in sorted_dataset:
            pids = np.array([ann["person_id"] for ann in dd["annotations"]])
            pids = pids[pids > -1].tolist()
            org_imgs_pids.append(pids)
            if len(pids) > max_p:
                max_p = len(pids)
        # padding
        for img_pids in org_imgs_pids:
            while len(img_pids) < max_p:
                img_pids.append(-2)
        g = torch.Generator()
        g.manual_seed(seed)
        assign_idxs = []
        for ti in range(times):
            all_imgs_idxs = list(range(size))  # indices to data dict
            len_diff = batch_size * nb - len(all_imgs_idxs)
            if len_diff > 0:
                append_idxs = torch.randint(0, size, (len_diff,), generator=g).tolist()
                all_imgs_idxs.extend(append_idxs)
            all_imgs_idxs = np.array(all_imgs_idxs)
            if shuffle:
                seq = torch.randperm(len(all_imgs_idxs), generator=g).tolist()
                all_imgs_idxs = all_imgs_idxs[seq]
            batch_idxs_built = all_imgs_idxs[:nb]
            batch_data_built = [
                [copy.deepcopy(org_imgs_pids[idx])] for idx in batch_idxs_built
            ]  # init batch
            batch_idxs_built = np.expand_dims(
                np.array(batch_idxs_built), axis=-1
            ).tolist()
            data_left_idxs = all_imgs_idxs[nb:].tolist()
            data_left = copy.deepcopy([org_imgs_pids[idx] for idx in data_left_idxs])
            for _ in range(batch_size - 1):
                cost_m = _assign_cost(data_left, batch_data_built, p, k)
                indices = linear_sum_assignment(cost_m)
                assigned_data_indices = []
                for i, j in zip(*indices):
                    assigned_data_indices.append(i)
                    batch_data_built[j].append(data_left[i])
                    batch_idxs_built[j].append(data_left_idxs[i])
                data_left_new = []
                data_left_idxs_new = []
                for di in range(len(data_left)):
                    if di not in assigned_data_indices:
                        data_left_new.append(data_left[di])
                        data_left_idxs_new.append(data_left_idxs[di])
                data_left = data_left_new
                data_left_idxs = data_left_idxs_new
            batch_idxs_seq = list(itertools.chain(*batch_idxs_built))
            assign_idxs.append(batch_idxs_seq)
        t_assign_idx = torch.tensor(assign_idxs, dtype=torch.int32)
        torch.save(t_assign_idx, save_path)
        comm.synchronize()


def _assign_cost(data_left, batch_built, p, k):
    data_left_arr = np.array(data_left)  # l x p
    batch_built_arr = np.array(batch_built)  # t x b x p
    fl_batch_built_arr = np.resize(
        batch_built_arr,
        (
            batch_built_arr.shape[0],
            batch_built_arr.shape[1] * batch_built_arr.shape[2],
        ),
    )  # t x bp
    ac = assign_cost_cuda.assign_cost(data_left_arr, fl_batch_built_arr, p, k)
    return ac


class APKSamplerPrec(Sampler):
    def __init__(
        self,
        size: int,
        pre_assign_path: str,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self._size = size
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._assign_idx = torch.load(pre_assign_path)
        self._shuffle = shuffle

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        all_epoch_idxs = self._assign_idx.numpy()
        if self._shuffle:
            seq = torch.randperm(len(self._assign_idx), generator=g).tolist()
            all_epoch_idxs = all_epoch_idxs[seq]
        next_epoch = 0
        while True:
            epoch_idx = all_epoch_idxs[next_epoch].tolist()
            yield from epoch_idx
            next_epoch = (next_epoch + 1) % all_epoch_idxs.shape[0]
