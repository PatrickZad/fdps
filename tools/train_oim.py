#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
OIM Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import sys

sys.path.append("./")
import logging
import os
from collections import OrderedDict
import torch

import psd2.utils.comm as comm
from psd2.checkpoint import DetectionCheckpointer
from psd2.config import get_cfg
from psd2.data import MetadataCatalog
from psd2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from psd2.evaluation import (
    InfDetEvaluator,
    QueryEvaluator,
    PrwQueryEvaluator,
    CuhkQueryEvaluator,
    CdpsQueryEvaluator,
    DatasetEvaluators,
    verify_results,
)
from psd2.modeling import GeneralizedRCNNWithTTA


class Trainer(DefaultTrainer):
    #     """
    #     Extension of the Trainer class adapted to SparseRCNN.
    #     """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        vis = cfg.TEST.VIS
        hist_only = cfg.TEST.VIS_HIST_ONLY
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type is "det":
            evaluator_list.append(
                InfDetEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    s_threds=[0.5],
                    vis=False,
                )
            )
        elif evaluator_type is "query":
            if "CUHK-SYSU" in dataset_name:
                evaluator_list.append(
                    CuhkQueryEvaluator(
                        dataset_name,
                        distributed=True,
                        output_dir=output_folder,
                        s_threds=[0.5],
                        vis=vis,
                        hist_only=hist_only,
                    )
                )
            elif "PRW" in dataset_name:
                evaluator_list.append(
                    PrwQueryEvaluator(
                        dataset_name,
                        distributed=True,
                        output_dir=output_folder,
                        s_threds=[0.5],
                        vis=vis,
                        hist_only=hist_only,
                    )
                )
            elif "CDPS" in dataset_name:
                evaluator_list.append(
                    CdpsQueryEvaluator(
                        dataset_name,
                        distributed=True,
                        output_dir=output_folder,
                        s_threds=[0.5],
                        vis=vis,
                        hist_only=hist_only,
                    )
                )
            else:
                evaluator_list.append(
                    QueryEvaluator(
                        dataset_name,
                        distributed=True,
                        output_dir=output_folder,
                        s_treds=[0.5],
                        vis=vis,
                        hist_only=hist_only,
                    )
                )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        from psd2.data.catalog import MapperCatalog
        from psd2.data.build import build_detection_train_loader

        mapper = MapperCatalog.get(cfg.DATASETS.TRAIN[0])(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        from psd2.data.catalog import MapperCatalog
        from psd2.data.build import get_detection_dataset_dicts, trivial_batch_collator
        from psd2.data.common import DatasetFromList, MapDataset
        from psd2.data.samplers import InferenceSampler
        import torch.utils.data as torchdata

        dataset = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=False,
            proposal_files=[
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)]
                for x in dataset_name
            ]
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        mapper = MapperCatalog.get(dataset_name)(cfg, is_train=False)
        # batched test
        if isinstance(dataset, list):
            dataset = DatasetFromList(dataset, copy=False)
        dataset = MapDataset(dataset, mapper)
        sampler = InferenceSampler(len(dataset))

        batch_sampler = torchdata.sampler.BatchSampler(
            sampler, cfg.TEST.IMS_PER_PROC, drop_last=False
        )
        data_loader = torchdata.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
        )
        return data_loader

    @classmethod
    def build_optimizer(cls, cfg, model):
        import itertools
        from psd2.solver.build import maybe_add_gradient_clipping

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        params = []
        for k, v in model.named_parameters():
            if v.requires_grad:
                if "BN" in k:
                    params += [{"params": [v], "lr": lr, "weight_decay": 0}]
                elif "bias" in k:
                    params += [{"params": [v], "lr": 2 * lr, "weight_decay": 0}]
                else:
                    params += [{"params": [v], "lr": lr, "weight_decay": weight_decay}]

        optimizer_type = cfg.MODEL.SEARCH.SOLVER.OPTIM
        if optimizer_type == "SGD":
            optimizer = maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "AdamW":
            optimizer = maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("psd2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
