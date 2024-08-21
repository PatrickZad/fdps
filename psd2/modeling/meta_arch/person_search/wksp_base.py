import itertools
import torch
from ..build import META_ARCH_REGISTRY
from .base import SearchBase
import psd2.utils.comm as comm
import logging
from psd2.utils.logger import log_every_n_seconds
import time
import datetime


@META_ARCH_REGISTRY.register()
class SearchBase(SearchBase):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.img_inst2pid_mem = None
        self.mem2pid = None

    @torch.no_grad()
    def label_update(self, data_loader):
        num_devices = comm.get_world_size()
        logger = logging.getLogger(__name__)
        logger.info("Start extraction on {} batches".format(len(data_loader)))

        total = len(data_loader)  # inference data loader must have a fixed length

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_data_time = 0
        total_compute_time = 0

        start_data_time = time.perf_counter()
        data_dicts = {}
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = self._inf_instances(inputs)
            # TODO outputs -> clustering input dict
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            total_seconds_per_iter = (
                time.perf_counter() - start_time
            ) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_iter * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Extraxtion: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total extraction time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(
            datetime.timedelta(seconds=int(total_compute_time))
        )
        logger.info(
            "Total extraction pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str,
                total_compute_time / (total - num_warmup),
                num_devices,
            )
        )
        logger.info("Gathering data on {} batches".format(len(data_loader)))
        comm.synchronize()
        all_data_dicts = comm.all_gather(data_dicts)
        all_data_dicts = itertools.chain(*all_data_dicts)
        for item in all_data_dicts:
            item["inst_feats"] = [feat.to(self.device) for feat in item["inst_feats"]]
        img_inst2pid_mem, mem2pid = {}, {}
        if comm.is_main_process():
            img_inst2pid_mem, mem2pid = self._clustering(all_data_dicts)
        comm.synchronize()
        sync_data = (img_inst2pid_mem, mem2pid)
        self.img_inst2pid_mem, self.mem2pid = comm.all_gather(sync_data)[0]
        new_memory = torch.stack(
            itertools.chain(*[item["inst_feats"] for item in all_data_dicts])
        )
        self.reid_head.update_memory(new_memory)

    def _clustering(self, data_dicts):
        """
        data_dicts:[
            {
                "image_id": image id,
                "inst_ids": instance indices in the image,
                "inst_feats": instance features
            },
            ...
        ]
        return:
            (img_id,inst_idx) -> (pseudo pid, memory idx) mapping,
            memory-idx -> pseudo-pid mapping
        """

        pass

    def _inf_instances(self, input_list):
        pass

    def get_pseudo_pid(self, img_id, inst_idx):
        return self.img_inst2pid_mem[img_id][inst_idx][0]

    def get_mem_idx(self, img_id, inst_idx):
        return self.img_inst2pid_mem[img_id][inst_idx][1]

    def get_pid_of_mem(self, mem_idx):
        return self.mem2pid[mem_idx]
