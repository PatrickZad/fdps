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
from multiprocessing.dummy import Pool as pool

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
        for dd in data_dicts:
            pids = np.array([ann["person_id"] for ann in dd["annotations"]])
            pids = pids[pids > -1].tolist()
            self.org_imgs_pids.append(pids)

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
        cost_func = functools.partial(self._assign_cost_single, batch_built=batch_built)
        """p = pool(8)
        many2many_cost = p.map(cost_func, data_left)
        p.close()
        p.join()"""
        many2many_cost = list(map(cost_func, data_left))
        return np.array(many2many_cost)

    def _assign_cost_single(self, img_pids, batch_built):
        cost_func = functools.partial(self._img2batch_cost, img_pids=img_pids)
        one2many_cost = list(map(cost_func, batch_built))
        return one2many_cost

    def _img2batch_cost(
        self,
        batch_pids,
        img_pids,
    ):
        merge_pids = copy.deepcopy(batch_pids)
        merge_pids.append(img_pids)
        b_pids = list(itertools.chain(*merge_pids))
        cnt = Counter(b_pids)
        cnt = dict(cnt)
        n_ids = [v for v in cnt.values()]
        n_ids = np.array(n_ids, dtype=np.float32)
        # p cost
        cost_p = (self.p - n_ids.shape[0]) ** 2
        # k cost
        if n_ids.shape[0] == 0:
            cost_k = (self.k) ** 2
        else:
            diff = n_ids - self.k
            cost_k = np.mean(diff * diff)
        return cost_p + cost_k
