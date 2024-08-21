# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from psd2.modeling.matcher import OlnHungarianMatcher
from ..build import META_ARCH_REGISTRY
from .ddetr_ps import DDETR_PS, SetCriterion


@META_ARCH_REGISTRY.register()
class DDETR_PS_Oln(DDETR_PS):
    """Use class_embbedding layers as centerness layers for simplicity"""

    @classmethod
    def from_config(cls, cfg):
        m_from_configs = cls.from_config(cfg)
        search_cfg = cfg.MODEL.SEARCH
        trans_cfg = search_cfg.D_TRANSFORMER
        ddetr_cfg = search_cfg.DDETR
        mt_cfg = search_cfg.MATCHER
        matcher = OlnHungarianMatcher(
            cost_score=mt_cfg.SET_COST_OLN,
            cost_bbox=mt_cfg.SET_COST_BBOX,
            cost_giou=mt_cfg.SET_COST_GIOU,
        )
        loss_w_cfg = search_cfg.LOSS_WEIGHTS
        weight_dict = {
            "loss_score": loss_w_cfg.OLN_LOSS,
            "loss_bbox": loss_w_cfg.BBOX_LOSS,
        }
        weight_dict["loss_giou"] = loss_w_cfg.GIOU_LOSS
        if ddetr_cfg.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(trans_cfg.DEC_DEPTH - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f"_enc": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        weight_dict["loss_oim"] = search_cfg.PERSON_FEAT.OIM.LOSS_WEIGHT
        if search_cfg.PERSON_FEAT.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(trans_cfg.DEC_DEPTH - 1):
                aux_weight_dict.update({"loss_oim" + f"_{i}": weight_dict["loss_oim"]})
            aux_weight_dict.update({"loss_oim" + f"_enc": weight_dict["loss_oim"]})
            weight_dict.update(aux_weight_dict)
        losses = ["scores", "boxes", "cardinality"]
        # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
        criterion = OlnSetCriterion(
            matcher,
            weight_dict,
            losses,
            search_cfg.OIM,
            focal_alpha=loss_w_cfg.FOCAL_ALPHA,
        )
        m_from_configs["matcher"] = matcher
        m_from_configs["criterion"] = criterion
        return m_from_configs


INF = 1e8
MINI = 1e-8
import functools


class OlnSetCriterion(SetCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        ps_cfg,
        focal_alpha=0.25,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__(1, matcher, weight_dict, losses, ps_cfg, focal_alpha)
        self.weight_dict.pop("class_error")

    def loss_scoress(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_ctns: Tensor = outputs["pred_logits"].sigmoid()  # b x n

        idx = self._get_src_permutation_idx(indices)
        score_match_cost = self.matcher._match_costs["score"]["cost"]
        pairwise_ct_bs = [
            cost.exp() * (-1) for cost in score_match_cost
        ]  # b-length list of n x gi
        target_ct_bs_o = torch.cat(
            [ct[i, j] for ct, (i, j) in zip(pairwise_ct_bs, indices)]
        )
        target_ctns = src_ctns.new_zeros(src_ctns.shape[:2])  # b x n
        target_ctns[idx] = target_ct_bs_o

        loss = F.l1_loss(src_ctns, target_ctns)
        return {"loss_score": loss}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "scores": self.loss_scoress,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "oim": self.loss_oim,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
