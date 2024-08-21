#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from psd2 import config
from ..build import META_ARCH_REGISTRY


from psd2.modeling.matcher import SrcnnHungarianMatcher as Matcher

from psd2.layers.set_criterion import EmaSrcnnSetCriterion as SetCriterion

from psd2.layers.extra_det_head import DynamicHead
from .sparse_rcnn_dcps import SparseRCNN_PS_DC
from ...backbone import build_backbone
import torch.nn as nn
import itertools


import torch


from torch import nn


from psd2.structures import Boxes


from psd2.modeling.matcher import SrcnnHungarianMatcher as Matcher
from psd2.layers.extra_det_head import DynamicHead

from psd2.structures.nested_tensor import NestedTensor
from psd2.config import configurable

from psd2.layers.set_criterion import EmaSrcnnSetCriterion as SetCriterion

from psd2.structures import Boxes
from psd2.utils.events import get_event_storage


@META_ARCH_REGISTRY.register()
class SparseRCNN_PS_DC_EMA(SparseRCNN_PS_DC):
    def _local_init(
        self, cfg, num_proposals, hidden_dim, in_features, criterion, use_focal
    ):
        super()._local_init(
            cfg, num_proposals, hidden_dim, in_features, criterion, use_focal
        )
        if self.training:
            self._ema_mm = cfg.MODEL.SEARCH.SRCNN.EMA_MM
            self._ema_init()

    def _ema_init(self):
        self.box_pooler_ema = DynamicHead._init_box_pooler(
            self.cfg, input_shape=self.backbone.output_shape()
        )
        self.backbone_ema = build_backbone(self.cfg)
        roi_size = self.cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        fpn_out_channels = self.cfg.MODEL.FPN.OUT_CHANNELS
        self.pfeat_head_ema = nn.Sequential(
            nn.Linear(
                roi_size ** 2 * fpn_out_channels,
                self.cfg.MODEL.SEARCH.PERSON_FEAT.FEAT_LEN,
            ),
            nn.BatchNorm1d(self.cfg.MODEL.SEARCH.PERSON_FEAT.FEAT_LEN),
        )
        train_params = itertools.chain(
            self.backbone.parameters(),
            self.head.box_pooler.parameters(),
            self.pfeat_head.parameters(),
        )
        ema_params = itertools.chain(
            self.backbone_ema.parameters(),
            self.box_pooler_ema.parameters(),
            self.pfeat_head_ema.parameters(),
        )
        for param_t, param_ema in zip(train_params, ema_params):
            param_ema.data.copy_(param_t.data)
            param_ema.requires_grad = False

    def _ema_update(self):
        train_params = itertools.chain(
            self.backbone.parameters(),
            self.head.box_pooler.parameters(),
            self.pfeat_head.parameters(),
        )
        ema_params = itertools.chain(
            self.backbone_ema.parameters(),
            self.box_pooler_ema.parameters(),
            self.pfeat_head_ema.parameters(),
        )
        for param_t, param_ema in zip(train_params, ema_params):
            param_ema.data = param_ema.data * self._ema_mm + param_t.data * (
                1 - self._ema_mm
            )

    @torch.no_grad()
    def _ema_output(self, batch_img, boxes_list):
        # Feature Extraction.
        src = self.backbone_ema(batch_img)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)
        rois_feats = self.box_pooler_ema(features, boxes_list)
        rois_feats = rois_feats.flatten(start_dim=1)
        psfeats = self.pfeat_head_ema(rois_feats)
        return psfeats

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

    def forward(self, input_list):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        if "query" in input_list[0]:
            return self.inf_query(input_list)
        input_batches = self.preprocess_input(input_list)
        img_nested_tensor: NestedTensor = input_batches[0]
        images_whwh = img_nested_tensor.tensors.new_tensor(
            [hw[::-1] * 2 for hw in img_nested_tensor.image_sizes]
        )
        targets = self.prepare_targets(input_batches)
        features, (outputs_class, outputs_coord) = self.get_det_pred(
            img_nested_tensor.tensors, images_whwh
        )

        # scale xyxy abs coords to [0,1] in aug_size
        aug_hw_bs = images_whwh.new_tensor(input_batches[5])  # B x 2
        aug_whwh_bs = torch.stack(
            [aug_hw_bs[:, 1], aug_hw_bs[:, 0]] * 2, dim=1
        )  # B x 4
        box_pooler = self.head.box_pooler
        nd, nb, nq, lb = outputs_coord.shape

        roi_boxes = outputs_coord.permute(1, 0, 2, 3).reshape(
            nb, -1, lb
        )  # B x (D x Nq) x 4

        if self.training and self.append_gt:
            # search loss makes no difference to learned boxes
            dt_roi_boxes = roi_boxes.clone().detach()
            boxes_list = [
                Boxes(torch.cat([dt_roi_boxes[bi], input_batches[3][bi]]))
                for bi in range(nb)
            ]  # append gt boxes
            rois_feats = box_pooler(features, boxes_list)  # (Bx[DxNq+g]) x 256 x 7 x 7
            rois_feats = rois_feats.flatten(start_dim=1)
            psfeats = self.pfeat_head(rois_feats)  # (Bx[DxNq+g]) x 256
            num_gts = [input_batches[3][bi].shape[0] for bi in range(nb)]
            num_qs = [nd * nq for _ in range(nb)]
            split_sizes = []
            for bi in range(nb):
                split_sizes.append(num_qs[bi])
                split_sizes.append(num_gts[bi])
            psfeats_splits = torch.split(psfeats, split_sizes)
            pred_psfeats = (
                torch.cat(psfeats_splits[::2], dim=0)
                .view(nb, nd, nq, -1)
                .permute(1, 0, 2, 3)
            )  # D x B x Nq x 256
            gt_psfeats = torch.cat(psfeats_splits[1::2], dim=0)
            ids_list = []
            for ids in zip(input_batches[4]):
                ids_list.append(
                    torch.tensor(ids, dtype=torch.int, device=self.device).squeeze(0)
                )
            gt_ids = torch.cat(ids_list, dim=-1).view(-1)
        else:
            boxes_list = [Boxes(roi_boxes[bi]) for bi in range(nb)]  # append gt boxes
            rois_feats = box_pooler(features, boxes_list)  # (BxDxNq) x 256 x 7 x 7
            rois_feats = rois_feats.flatten(start_dim=1)
            pred_psfeats = self.pfeat_head(rois_feats)  # (BxDxNq) x 256
            pred_psfeats = pred_psfeats.view(nb, nd, nq, -1).permute(
                1, 0, 2, 3
            )  # D x B x Nq x 256

        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "reid_feats": pred_psfeats[-1],
        }  # xyxy boxes in aug [0,1]

        if self.training:
            if self.deep_supervision:
                output["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b, "reid_feats": c}
                    for a, b, c in zip(
                        outputs_class[:-1], outputs_coord[:-1], pred_psfeats[:-1]
                    )
                ]
            # update ema
            self._ema_update()
            # ema gts
            ema_gt_boxes_list = [
                Boxes(input_batches[3][bi]) for bi in range(nb)
            ]  # gt boxes
            ema_gt_feats = self._ema_output(
                img_nested_tensor.tensors, ema_gt_boxes_list
            )
            ema_ids_list = []
            for ids in zip(input_batches[4]):
                ema_ids_list.append(
                    torch.tensor(ids, dtype=torch.int, device=self.device).squeeze(0)
                )
            ema_gt_ids = torch.cat(ema_ids_list, dim=-1).view(-1)
            ema_feats_ids = {"ema_reid_feats": ema_gt_feats, "ema_ids": ema_gt_ids}
            if self.append_gt:
                loss_dict = self.criterion(
                    output,
                    targets,
                    ema_feats_ids=ema_feats_ids,
                    gt_feats_ids={"reid_feats": gt_psfeats, "ids": gt_ids},
                )
            else:
                loss_dict = self.criterion(output, targets, ema_feats_ids=ema_feats_ids)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            if get_event_storage().iter % self.vis_period == 0:
                self.visualize_training(
                    input_batches,
                    features,
                    output,
                    list(gt_psfeats.split(num_gts)),
                    ids_list,
                )
            return loss_dict

        else:
            aug_hws = output["pred_boxes"].new_tensor(input_batches[5])
            org_hws = output["pred_boxes"].new_tensor(input_batches[6])  # B x 2
            logits = output.pop("pred_logits", None)
            output["pred_scores"] = logits.sigmoid()
            org_whwh = torch.stack([org_hws[:, 1], org_hws[:, 0]] * 2, dim=1)  # B x 4
            aug_whwh = torch.stack([aug_hws[:, 1], aug_hws[:, 0]] * 2, dim=1)  # B x 4
            sf = org_whwh / aug_whwh
            # back to org abs
            output["pred_boxes"] *= sf.unsqueeze(1)  # B x 1 x 4

            inter_outs = output.pop("aux_outputs", None)
            return output
