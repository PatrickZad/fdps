#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from psd2 import config
from ..build import META_ARCH_REGISTRY


from psd2.modeling.matcher import SrcnnHungarianMatcher as Matcher


from psd2.layers.extra_det_head import DynamicHead
from .sparse_rcnn_dcps_ema import SparseRCNN_PS_DC_EMA
from ...backbone import build_backbone
import torch.nn as nn
import itertools


from psd2.modeling.matcher import SrcnnHungarianMatcher as Matcher
from psd2.layers.extra_det_head import DynamicHead


from psd2.layers.set_criterion import EmaTridSrcnnSetCriterion as SetCriterion


@META_ARCH_REGISTRY.register()
class SparseRCNN_PS_DC_EMA_TRID(SparseRCNN_PS_DC_EMA):
    @classmethod
    def from_config(cls, cfg):
        search_cfg = cfg.MODEL.SEARCH
        srcnn_cfg = search_cfg.SRCNN
        rcnn_head_cfg = srcnn_cfg.RCNN_HEAD
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        num_proposals = srcnn_cfg.NUM_PROPOSALS
        hidden_dim = rcnn_head_cfg.HIDDEN_DIM
        num_heads = rcnn_head_cfg.NUM_HEADS

        # Loss parameters:
        loss_cfg = search_cfg.LOSS_WEIGHTS
        class_weight = loss_cfg.CLASS_WEIGHT
        giou_weight = loss_cfg.GIOU_WEIGHT
        l1_weight = loss_cfg.L1_WEIGHT
        no_object_weight = loss_cfg.NO_OBJECT_WEIGHT
        deep_supervision = loss_cfg.DEEP_SUPERVISION
        use_focal = loss_cfg.FOCAL.USE_FOCAL

        # Build Criterion.
        matcher = Matcher(
            cfg=cfg,
            cost_class=class_weight,
            cost_bbox=l1_weight,
            cost_giou=giou_weight,
            use_focal=use_focal,
        )
        weight_dict = {
            "loss_ce": class_weight,
            "loss_bbox": l1_weight,
            "loss_giou": giou_weight,
        }
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        weight_dict["loss_oim"] = search_cfg.PERSON_FEAT.OIM.LOSS_WEIGHT
        if search_cfg.PERSON_FEAT.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(num_heads - 1):
                aux_weight_dict.update({"loss_oim" + f"_{i}": weight_dict["loss_oim"]})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes"]

        criterion = SetCriterion(
            cfg=cfg,
            num_classes=1,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            use_focal=use_focal,
        )
        return {
            "cfg": cfg,
            "num_proposals": num_proposals,
            "hidden_dim": hidden_dim,
            "in_features": in_features,
            "criterion": criterion,
            "use_focal": use_focal,
        }
