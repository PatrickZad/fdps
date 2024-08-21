#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in psd2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use psd2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import sys

sys.path.append("./")
import logging
import os
from collections import OrderedDict
import torch
import wandb
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


def build_evaluator(cfg, dataset_name, output_folder=None):
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
                s_threds=[0.2],
                vis=vis,
            )
        )
    elif evaluator_type is "query":
        if "CUHK-SYSU" in dataset_name:
            evaluator_list.append(
                CuhkQueryEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    s_threds=[0.2],
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
                    s_threds=[0.2],
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
                    s_threds=[0.2],
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
                    s_treds=[0.2, 0.3, 0.5],
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


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    """
    def __init__(self, cfg, model_find_unused_parameters):
        super().__init__(cfg, model_find_unused_parameters=model_find_unused_parameters)
        # TODO use wandb to visualize gradients
        self.log_wandb = False
        if comm.is_main_process():
            wb_cfg = cfg.EXT_VIS.W_AND_B
            if wb_cfg.VIS_GRADIENT:
                proj = wb_cfg.PROJECT
                entity = wb_cfg.ENTITY
                cfg_dict = cfg.__dict__
                wandb.init(project=proj, entity=entity)
                wandb.config = cfg_dict
                wandb.watch(self._trainer.model, log_freq=100)
                self.log_wandb = True
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

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

    @classmethod
    def build_optimizer(cls, cfg, model):
        from psd2.solver.build import maybe_add_gradient_clipping

        optim_name = cfg.DETECTOR.SOLVER.OPTIM
        lr_backbone_names = ["backbone"]
        lr_offset_names = ["sampling_offsets"]  # MSDAttn
        lr_ref_pts_names = ["reference_points"]  # linear on obj quries

        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not _match_name_keywords(n, lr_backbone_names)
                    and not _match_name_keywords(n, lr_offset_names)
                    and not _match_name_keywords(n, lr_ref_pts_names)
                    and p.requires_grad
                ],
                "lr": cfg.SOLVER.BASE_LR,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if _match_name_keywords(n, lr_backbone_names) and p.requires_grad
                ],
                "lr": cfg.SOLVER.BASE_LR * cfg.DETECTOR.SOLVER.BACKBONE_LR_FACTOR,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if _match_name_keywords(n, lr_offset_names) and p.requires_grad
                ],
                "lr": cfg.SOLVER.BASE_LR * cfg.DETECTOR.SOLVER.SAMP_OFFSETS_LR_FACTOR,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if _match_name_keywords(n, lr_ref_pts_names) and p.requires_grad
                ],
                "lr": cfg.SOLVER.BASE_LR * cfg.DETECTOR.SOLVER.REF_POINTS_LR_FACTOR,
            },
        ]
        if optim_name is "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                param_dicts,
                lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        assert optim_name in ("Adam", "AdamW")
        return maybe_add_gradient_clipping(cfg, getattr(torch.optim, optim_name))(
            param_dicts,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

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


def _match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


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

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg, model_find_unused_parameters=not args.n_find_unparams)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
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
