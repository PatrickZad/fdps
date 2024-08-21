import torch
from psd2.modeling.poolers import ROIPooler
import torch.nn as nn
from .roi_coseg_head import RoiCoSegHead
from psd2.config import configurable
from psd2.layers import Conv2d


class MsRoiCoSegHead(RoiCoSegHead):
    def _append_init(self, head_cfg):
        self.out_conv = Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=torch.nn.BatchNorm2d(self.in_channels),
        )

    @classmethod
    def from_config(cls, cfg):
        in_channels = cfg.IN_CHANNELS
        pooler_cfg = cfg.ROI_POOLER
        roi_size = pooler_cfg.POOLER_RESOLUTION
        roi_scales = pooler_cfg.POOLER_SCALES
        samp_ratio = pooler_cfg.POOLER_SAMPLING_RATIO
        pooler_type = pooler_cfg.POOLER_TYPE
        box_poolers = nn.ModuleList(
            [
                ROIPooler(
                    output_size=roi_size,
                    scales=[sc],
                    sampling_ratio=samp_ratio,
                    pooler_type=pooler_type,
                )
                for sc in roi_scales
            ]
        )
        return {
            "head_cfg": cfg,
            "in_channels": in_channels,
            "roi_size": roi_size,
            "box_pooler": box_poolers,
        }

    def get_roi_feats(self, bk_feats, d2_boxes):
        boxes_rois = []
        for fi, box_pooler in enumerate(self.box_pooler):
            boxes_rois.append(
                box_pooler(bk_feats[fi : fi + 1], d2_boxes)
            )  # (B x Nq) x 256 x 24 x 12
        return self.out_conv(
            torch.cat(boxes_rois, dim=1)
        )  # (B x Nq) x (256 x p) x 24 x 12
