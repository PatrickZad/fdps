

import torch


from psd2.modeling.poolers import ROIPooler
from psd2.structures import Boxes
import torch.nn as nn
from torch.nn import init

from .roi_fc_head import RoiFcHead
from psd2.layers import Conv2d
from psd2.layers.pooling import *
from typing import Iterable

import itertools


class MsRoiFcHead(RoiFcHead):
    def _init_modules(
        self, roi_size, roi_scales, samp_ratio, roipooler_type, pool_layer
    ):
        self.box_poolers = nn.ModuleList(
            [
                ROIPooler(
                    output_size=roi_size,
                    scales=[sc],
                    sampling_ratio=samp_ratio,
                    pooler_type=roipooler_type,
                )
                for sc in roi_scales
            ]
        )
        self.feat_bn = nn.BatchNorm1d(self.pfeat_dim)

        init.normal_(self.feat_bn.weight, std=0.01)
        init.constant_(self.feat_bn.bias, 0)

        if pool_layer == "fastavgpool":
            self.feat_proj = FastGlobalAvgPool2d()
        elif pool_layer == "avgpool":
            self.feat_proj = nn.AdaptiveAvgPool2d(1)
        elif pool_layer == "maxpool":
            self.feat_proj = nn.AdaptiveMaxPool2d(1)
        elif pool_layer == "gempoolP":
            self.feat_proj = GeneralizedMeanPoolingP()  # this one
        elif pool_layer == "gempool":
            self.feat_proj = GeneralizedMeanPooling()
        elif pool_layer == "avgmaxpool":
            self.feat_proj = AdaptiveAvgMaxPool2d()
        elif pool_layer == "clipavgpool":
            self.feat_proj = ClipGlobalAvgPool2d()
        elif pool_layer == "identity":
            self.feat_proj = nn.Identity()
        elif pool_layer == "flatten":
            self.feat_proj = Flatten()
        elif pool_layer == "linear":
            self.feat_proj = FlattenLinear(
                roi_size[0] * roi_size[1] * self.in_channels,
                self.pfeat_dim,
            )
        else:
            raise KeyError(f"{pool_layer} is not supported!")
        self.out_conv = Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=torch.nn.BatchNorm2d(self.in_channels),
        )

    def get_roi_feats(self, bk_feats, d2_boxes):
        boxes_rois = []
        for fi, box_pooler in enumerate(self.box_poolers):
            boxes_rois.append(
                box_pooler(bk_feats[fi : fi + 1], d2_boxes)
            )  # (B x Nq) x 256 x 24 x 12
        roi_feats = torch.cat(boxes_rois, dim=1)
        return self.out_conv(roi_feats)  # (B x Nq) x (256 x p) x 24 x 12


class MsRoiFcHeadSide(RoiFcHead):
    def __init__(self, head_cfg):
        super().__init__(head_cfg)
        self.side_pretrain = head_cfg.PRETAIN_SIDE_NET

    def _init_modules(
        self, roi_size, roi_scales, samp_ratio, roipooler_type, pool_layer
    ):
        self.box_poolers = nn.ModuleList(
            [
                ROIPooler(
                    output_size=roi_size,
                    scales=[sc],
                    sampling_ratio=samp_ratio,
                    pooler_type=roipooler_type,
                )
                for sc in roi_scales
            ]
        )
        self.feat_bn = nn.BatchNorm1d(self.pfeat_dim)
        self.feat_bn_side = nn.BatchNorm1d(self.pfeat_dim)

        init.normal_(self.feat_bn.weight, std=0.01)
        init.constant_(self.feat_bn.bias, 0)
        init.normal_(self.feat_bn_side.weight, std=0.01)
        init.constant_(self.feat_bn_side.bias, 0)

        if pool_layer == "fastavgpool":
            self.feat_proj = FastGlobalAvgPool2d()
            self.feat_proj_side = FastGlobalAvgPool2d()
        elif pool_layer == "avgpool":
            self.feat_proj = nn.AdaptiveAvgPool2d(1)
            self.feat_proj_side = nn.AdaptiveAvgPool2d(1)
        elif pool_layer == "maxpool":
            self.feat_proj = nn.AdaptiveMaxPool2d(1)
            self.feat_proj_side = nn.AdaptiveMaxPool2d(1)
        elif pool_layer == "gempoolP":
            self.feat_proj = GeneralizedMeanPoolingP()  # this one
            self.feat_proj_side = GeneralizedMeanPoolingP()  # this one
        elif pool_layer == "gempool":
            self.feat_proj = GeneralizedMeanPooling()
            self.feat_proj_side = GeneralizedMeanPooling()
        elif pool_layer == "avgmaxpool":
            self.feat_proj = AdaptiveAvgMaxPool2d()
            self.feat_proj_side = AdaptiveAvgMaxPool2d()
        elif pool_layer == "clipavgpool":
            self.feat_proj = ClipGlobalAvgPool2d()
            self.feat_proj_side = ClipGlobalAvgPool2d()
        elif pool_layer == "identity":
            self.feat_proj = nn.Identity()
            self.feat_proj_side = nn.Identity()
        elif pool_layer == "flatten":
            self.feat_proj = Flatten()
            self.feat_proj_side = Flatten()
        elif pool_layer == "linear":
            self.feat_proj = FlattenLinear(
                roi_size[0] * roi_size[1] * self.in_channels,
                self.pfeat_dim,
            )
            self.feat_proj_side = FlattenLinear(
                roi_size[0] * roi_size[1] * self.in_channels,
                self.pfeat_dim,
            )
        else:
            raise KeyError(f"{pool_layer} is not supported!")
        self.side_alpha = nn.Parameter(torch.tensor(0.0))
        self.out_conv = Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=torch.nn.BatchNorm2d(self.in_channels),
        )
        self.out_conv_side = Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=torch.nn.BatchNorm2d(self.in_channels),
        )

    def get_roi_feats(self, bk_feats, d2_boxes):
        boxes_rois = []
        for fi, box_pooler in enumerate(self.box_poolers):
            boxes_rois.append(
                box_pooler(bk_feats[fi : fi + 1], d2_boxes)
            )  # (B x Nq) x 256 x 24 x 12
        roi_feats = torch.cat(boxes_rois, dim=1)
        return self.out_conv(roi_feats), self.out_conv_side(
            roi_feats
        )  # (B x Nq) x (256 x p) x 24 x 12

    def get_pfeats(self, bk_feats, roi_boxes):
        if isinstance(roi_boxes, torch.Tensor):
            bs, qs = roi_boxes.shape[:2]
            boxes_list = [Boxes(roi_boxes[bi]) for bi in range(bs)]
            rois_feats, rois_feats_side = self.get_roi_feats(
                bk_feats, boxes_list
            )  # (B x Nq) x 256 x 24 x 12
            mpfeats, cpfeats = self.pfeat_head(rois_feats, rois_feats_side)
            return (
                mpfeats.reshape(bs, qs, -1),
                cpfeats.reshape(bs, qs, -1),
                rois_feats.reshape(bs, qs, *rois_feats.shape[-3:]),
            )

        else:
            assert isinstance(roi_boxes, Iterable)
            num_splits = [boxes.shape[0] for boxes in roi_boxes]
            bs = len(roi_boxes)
            boxes_list = [Boxes(roi_boxes[bi]) for bi in range(bs)]
            rois_feats, rois_feats_side = self.get_roi_feats(
                bk_feats, boxes_list
            )  # ( b0 + b1+ ...) x 256 x 24 x 8
            mpfeats, cpfeats = self.pfeat_head(rois_feats, rois_feats_side)
            return (
                torch.split(mpfeats, num_splits),
                torch.split(cpfeats, num_splits),
                torch.split(rois_feats, num_splits),
            )

    def pfeat_head(self, rois_feats, rois_feats_side):
        alpha_weight = torch.sigmoid(self.side_alpha)
        proj_feats = self.feat_proj(rois_feats)
        proj_feats_side = self.feat_proj_side(rois_feats_side)
        if proj_feats.dim() > 2:
            proj_feats = proj_feats.reshape(proj_feats.shape[0], proj_feats.shape[1])
        if proj_feats_side.dim() > 2:
            proj_feats_side = proj_feats_side.reshape(
                proj_feats_side.shape[0], proj_feats_side.shape[1]
            )
        after_bn_feat = self.feat_bn(proj_feats)
        after_bn_feat_side = self.feat_bn_side(proj_feats_side)
        if self.training:
            feat_at = self.train_feat_at
        else:
            feat_at = self.inf_feat_at
        if feat_at == "after_bn":
            cls_feats = (
                alpha_weight * after_bn_feat + (1 - alpha_weight) * after_bn_feat_side
            )
        elif feat_at == "before_bn":
            cls_feats = alpha_weight * proj_feats + (1 - alpha_weight) * proj_feats_side
        else:
            raise KeyError(f"{feat_at} is not supported!")
        if self.metric_feat_at == "after_bn":
            metric_feats = (
                alpha_weight * after_bn_feat + (1 - alpha_weight) * after_bn_feat_side
            )
        elif self.metric_feat_at == "before_bn":
            metric_feats = (
                alpha_weight * proj_feats + (1 - alpha_weight) * proj_feats_side
            )
        else:
            raise KeyError(f"{self.metric_feat_at} is not supported!")
        return metric_feats, cls_feats



            