import torch
from torch.functional import Tensor
from .base_reid_head import ReidHeadBase
from psd2.modeling.poolers import ROIPooler
from psd2.structures import Boxes
import torch.nn as nn
from torch.nn import init
from psd2.layers.pooling import *
from psd2.layers.mem_matching_losses import build_loss_layer
import copy
from psd2.layers.shaping_loss import shapingloss
from psd2.utils.events import get_event_storage
import numpy as np
import colorsys
import torchvision.transforms.functional as tvF
import torch.nn.functional as tF
from psd2.config import configurable
from typing import Iterable
import cv2
import time
import os

INF = 1e8


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# Bottleneck of standard ResNet50/101, with kernel size equal to 1
class Bottleneck1x1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1x1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RoiCoSegHead(ReidHeadBase):
    @configurable
    def __init__(self, head_cfg, in_channels, roi_size, box_pooler):
        super().__init__(head_cfg)

        ################ box poolxer ################

        self.box_pooler = box_pooler
        self.box_gradient = head_cfg.BOX_BP

        ################ global feat pooler ################
        pool_type = head_cfg.POOL_LAYER
        if pool_type == "fastavgpool":
            self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == "avgpool":
            self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "maxpool":
            self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == "gempoolP":
            self.pool_layer = GeneralizedMeanPoolingP()  # this one
        elif pool_type == "gempool":
            self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":
            self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "clipavgpool":
            self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":
            self.pool_layer = nn.Identity()
        elif pool_type == "flatten":
            self.pool_layer = Flatten()
        elif pool_type == "linear":
            self.pool_layer = FlattenLinear(
                roi_size[0] * roi_size[1] * in_channels,
                self.pfeat_dim,
            )
        else:
            raise KeyError(f"{pool_type} is not supported!")

        ################ global bottelneck ################
        # TODO check if init matters
        self.bottleneck_global = nn.BatchNorm1d(self.pfeat_dim)
        init.normal_(self.bottleneck_global.weight, std=0.01)
        init.constant_(self.bottleneck_global.bias, 0)
        self.train_feat_at = head_cfg.PERSON_FEATURE.TRAIN_AT
        self.inf_feat_at = head_cfg.PERSON_FEATURE.INF_AT

        ################ part feat ################
        self.num_parts = head_cfg.PERSON_FEATURE.PART.NUM_PARTS
        self.part_dim = head_cfg.PERSON_FEATURE.PART.DIM
        self.grouping = GroupingUnit(in_channels, self.num_parts)
        self.grouping.reset_parameters(init_weight=None, init_smooth_factor=None)
        # post-processing bottleneck block for the region features
        # TODO more channels
        self.post_block = nn.Sequential(
            Bottleneck1x1(
                in_channels,
                in_channels // 4,
                stride=1,
                downsample=nn.Sequential(
                    nn.Conv2d(
                        in_channels, in_channels, kernel_size=1, stride=1, bias=False
                    ),
                    nn.BatchNorm2d(in_channels),
                ),
            ),
            Bottleneck1x1(in_channels, in_channels // 4, stride=1),
            Bottleneck1x1(in_channels, in_channels // 4, stride=1),
            Bottleneck1x1(in_channels, in_channels // 4, stride=1),
        )
        self.decrease_dim_block = nn.Sequential(  # pcb
            nn.Conv2d(in_channels, self.part_dim, 1, 1, bias=False),  #
            nn.BatchNorm2d(self.part_dim),
            nn.ReLU(inplace=True),
        )

        ################ part bottelneck ################
        # the final batchnorm
        # self.groupingbn = nn.BatchNorm2d(512 * 4)
        feat_dim_part = self.part_dim * self.num_parts
        self.bottleneck_part = BatchNorm(feat_dim_part, bias_freeze=True)

        ################ initilization ################
        # apply on grouping, post_block, decrease_dim_block, bottleneck_part
        # initialize convolutional layers with kaiming_normal_, BatchNorm with weight 1, bias 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize the last bn in residual blocks with weight zero
        for m in self.modules():
            if isinstance(m, Bottleneck1x1):
                nn.init.constant_(m.bn3.weight, 0)

        # global f
        # self.bottleneck.apply(weights_init_kaiming)
        # part f
        self.bottleneck_part.apply(weights_init_kaiming)

        ################ final feat ################
        self.feat_type = head_cfg.PERSON_FEATURE.FEATURE_TYPE

        ################ roi memories ################
        # save to aviod duplicate roi extraction
        self.rois_feats = None
        ################ part losses ################
        loss_cfg = head_cfg.LOSS
        loss_layer_part = build_loss_layer(loss_cfg, feat_dim_part)
        if loss_cfg.AUX_LOSS > 0:
            self.part_loss_layers = _get_clones(loss_layer_part, loss_cfg.AUX_LOSS)
            self.part_loss_layers.append(loss_layer_part)
        else:
            self.part_loss_layers = nn.ModuleList([loss_layer_part])
        # non-overlap loss
        sl_loss_cfg = loss_cfg.SHAPING_LOSS
        self.sl_std = sl_loss_cfg.STD
        self.sl_radius = sl_loss_cfg.RADIUS
        self.sl_alpha = sl_loss_cfg.ALPHA
        self.sl_beta = sl_loss_cfg.BETA
        self.sl_eps = sl_loss_cfg.EPS
        self.sl_loss_weight = loss_cfg.LOSS_WEIGHTS.NON_LAP
        # visualization
        self.vis_period = head_cfg.VIS_PERIOD
        self.vis_inf = head_cfg.VIS_INF
        self.vis_inf_dir = head_cfg.VIS_INF_SAVE
        if self.vis_inf and not os.path.exists(self.vis_inf_dir):
            os.makedirs(self.vis_inf_dir)

        self._append_init(head_cfg)

    def _append_init(self, head_cfg):
        pass

    @classmethod
    def from_config(cls, cfg):
        in_channels = cfg.IN_CHANNELS
        pooler_cfg = cfg.ROI_POOLER
        roi_size = pooler_cfg.POOLER_RESOLUTION

        box_pooler = ROIPooler(
            output_size=roi_size,
            scales=pooler_cfg.POOLER_SCALES,
            sampling_ratio=pooler_cfg.POOLER_SAMPLING_RATIO,
            pooler_type=pooler_cfg.POOLER_TYPE,
        )
        return {
            "head_cfg": cfg,
            "in_channels": in_channels,
            "roi_size": roi_size,
            "box_pooler": box_pooler,
        }

    def get_pfeats(self, bk_feats, roi_boxes):
        # global feat
        if isinstance(roi_boxes, torch.Tensor):
            bs, qs = roi_boxes.shape[:2]
            boxes_list = [Boxes(roi_boxes[bi]) for bi in range(bs)]
            rois_feats = self.get_roi_feats(
                bk_feats, boxes_list
            )  # (B x Nq) x 256 x 24 x 12
            mpfeats, cpfeats = self.pfeat_head_global(rois_feats)
            self.rois_feats = rois_feats.view(bs, qs, *rois_feats.shape[-3:])
            return mpfeats.reshape(bs, qs, -1), cpfeats.reshape(bs, qs, -1)

        else:
            assert isinstance(roi_boxes, Iterable)
            num_splits = [boxes.shape[0] for boxes in roi_boxes]
            bs = len(roi_boxes)
            boxes_list = [Boxes(roi_boxes[bi]) for bi in range(bs)]
            rois_feats = self.get_roi_feats(
                bk_feats, boxes_list
            )  # ( b0 + b1+ ...) x 256 x 24 x 8
            mpfeats, cpfeats = self.pfeat_head_global(rois_feats)
            self.rois_feats = torch.split(rois_feats, num_splits)
            return torch.split(mpfeats, num_splits), torch.split(cpfeats, num_splits)

    def get_pfeats_part(self):
        # part feat
        rois_feats = self.rois_feats
        assert rois_feats is not None
        if isinstance(self.rois_feats, torch.Tensor):
            # B x Nq x 256 x 24 x 12
            bs, qs = rois_feats.shape[:2]
            rois_feats = rois_feats.flatten(0, 1)
            mpfeats, cpfeats, assign = self.pfeat_head_part(rois_feats)
            ac, ah, aw = assign.shape[-3:]
            cpfeats = cpfeats.reshape((bs, qs, -1))
            mpfeats = mpfeats.reshape((bs, qs, -1))
            assign = assign.reshape((bs, qs, ac, ah, aw))
        else:
            assert isinstance(self.rois_feats, Iterable)
            # ( b0 + b1+ ...) x 256 x 24 x 8
            num_rois = [roi_feat.shape[0] for roi_feat in rois_feats]
            rois_feats = torch.cat(rois_feats, dim=0)
            mpfeats, cpfeats, assign = self.pfeat_head_part(rois_feats)
            cpfeats = torch.split(cpfeats, num_rois)
            mpfeats = torch.split(mpfeats, num_rois)
            assign = torch.split(assign, num_rois)
        self.rois_feats = None
        return mpfeats, cpfeats, assign

    def pfeat_head_global(self, rois_feats):
        # global feat
        global_feat = self.pool_layer(rois_feats)
        if global_feat.dim() > 2:
            global_feat = global_feat.reshape(
                global_feat.shape[0], global_feat.shape[1]
            )
        bn_feat = self.bottleneck_global(global_feat)
        neck_feat = self.train_feat_at if self.training else self.inf_feat_at
        if neck_feat == "before_bn":
            feat = global_feat.view(global_feat.shape[0], global_feat.shape[1])
        elif neck_feat == "after_bn":
            feat = bn_feat.view(bn_feat.shape[0], bn_feat.shape[1])
        else:
            raise KeyError(f"{neck_feat} is invalid !")
        if self.metric_feat_at == "after_bn":
            metric_feats = bn_feat.view(bn_feat.shape[0], bn_feat.shape[1])
        elif self.metric_feat_at == "before_bn":
            metric_feats = global_feat.view(global_feat.shape[0], global_feat.shape[1])
        else:
            raise KeyError(f"{self.metric_feat_at} is invalid!")

        return metric_feats, feat

    def pfeat_head_part(self, rois_feats):

        # inter
        # grouping module upon the feature maps outputed by the backbone
        region_feature, assign = self.grouping(
            rois_feats
        )  # B x 256 x 5, B x 5 x 24 x 8
        region_feature = region_feature.contiguous().unsqueeze(3)  # B x 256 x 5 x 1

        # non-linear layers over the region features -- GNN
        region_feature = self.post_block(region_feature)  # B x 256 x 5 x 1

        part_feat = self.decrease_dim_block(region_feature)  # B x 128 x 5 x 1
        bn_part_feat = part_feat.contiguous().view(
            region_feature.size(0), -1
        )  # cat, B x (128 x 5)

        neck_feat = self.train_feat_at if self.training else self.inf_feat_at
        if neck_feat == "before_bn":
            # TODO check if correct
            feat_part = part_feat[..., 0, 0]
        elif neck_feat == "after_bn":
            feat_part = bn_part_feat
        else:
            raise KeyError(f"{neck_feat} is invalid !")
        if self.metric_feat_at == "after_bn":
            metric_feats = bn_part_feat
        elif self.metric_feat_at == "before_bn":
            metric_feats = bn_part_feat
        else:
            raise KeyError(f"{self.metric_feat_at} is not supported!")

        return metric_feats, feat_part, assign

    def feats_fuse(self, b_global_feats, b_part_feats):
        if isinstance(b_global_feats, torch.Tensor):
            assert torch.all(
                torch.tensor(b_global_feats.shape[:-1])
                == torch.tensor(b_part_feats.shape[:-1])
            )
            if self.feat_type == "cat":
                return tF.normalize(
                    torch.cat([b_global_feats, b_part_feats], dim=-1), dim=-1
                )
            elif self.feat_type == "global":
                return tF.normalize(b_global_feats, dim=-1)
            elif self.feat_type == "local":
                return tF.normalize(b_part_feats, dim=-1)
            else:
                raise KeyError(f"{self.feat_type} is not supported!")
        else:
            assert isinstance(b_global_feats, Iterable)
            bs = len(b_global_feats)
            fuse_feats = []
            for bi in range(bs):
                if self.feat_type == "cat":
                    fuse_feats.append(
                        tF.normalize(
                            torch.cat([b_global_feats[bi], b_part_feats[bi]], dim=-1),
                            dim=-1,
                        )
                    )
                elif self.feat_type == "global":
                    fuse_feats.append(tF.normalize(b_global_feats[bi], dim=-1))
                elif self.feat_type == "local":
                    fuse_feats.append(tF.normalize(b_part_feats[bi], dim=-1))
                else:
                    raise KeyError(f"{self.feat_type} is not supported!")
            return fuse_feats

    def forward(self, bk_feats, det_outputs, targets, det_match_indices):
        head_outputs = {}
        if self.training:
            if self.append_gt:
                assert targets is not None
            head_outputs["losses"] = {}
            if len(self.part_loss_layers) > 1:
                # TODO metric feats and losses
                assert "aux_outputs" in det_outputs
                inter_outs = det_outputs["aux_outputs"] + [
                    {
                        "pred_logits": det_outputs["pred_logits"],
                        "pred_boxes": det_outputs["pred_boxes"],
                    }
                ]
                head_aux_outputs = []
                bs = inter_outs[0]["pred_boxes"].shape[0]
                lvl_bn_ids, lvl_bn_asc, lvl_bn_boxes, lvl_bn_logits = [], [], [], []
                lvl_bn_nums = []
                gt_ids, gt_asc, gt_boxes, gt_logits = self.get_gt_id_asc_box_logits(
                    targets
                )

                for i in range(len(inter_outs)):
                    (
                        assign_ids,
                        bn_ids,
                        bn_asc,
                        bn_boxes,
                        bn_logits,
                    ) = self.split_id_asc_box_logits(
                        inter_outs[i], targets, det_match_indices[i]
                    )  # (b1, b2, ...) x t

                    if self.append_gt:
                        for bi in range(bs):
                            bn_ids[bi] = torch.cat([bn_ids[bi], gt_ids[bi]], dim=0)
                            bn_asc[bi] = torch.cat([bn_asc[bi], gt_asc[bi]], dim=0)
                            bn_boxes[bi] = torch.cat(
                                [bn_boxes[bi], gt_boxes[bi]], dim=0
                            )
                            bn_logits[bi] = torch.cat(
                                [bn_logits[bi], gt_logits[bi]], dim=0
                            )
                    lvl_bn_ids.append(bn_ids)
                    lvl_bn_asc.append(bn_asc)
                    lvl_bn_boxes.append(bn_boxes)
                    lvl_bn_logits.append(bn_logits)
                    lvl_bn_nums.append([ids.shape[0] for ids in bn_ids])
                    if i < len(inter_outs) - 1:
                        head_aux_outputs.append({"assign_ids": assign_ids})
                    else:
                        head_outputs["assign_ids"] = assign_ids
                bn_ids, bn_asc, bn_boxes, bn_logits = [], [], [], []
                for bi in range(bs):
                    bn_ids.append(torch.cat([l_ids[bi] for l_ids in lvl_bn_ids], dim=0))
                    bn_asc.append(torch.cat([l_asc[bi] for l_asc in lvl_bn_asc], dim=0))
                    bn_boxes.append(
                        torch.cat([l_boxes[bi] for l_boxes in lvl_bn_boxes], dim=0)
                    )
                    bn_logits.append(
                        torch.cat([l_logits[bi] for l_logits in lvl_bn_logits], dim=0)
                    )
                bn_reid_feats_g = self.get_pfeats(bk_feats, bn_boxes)  # bn x lvl
                bn_reid_feats_p, bn_part_assign = self.get_pfeats_part()
                lvl_bn_pfeats_g = [[]] * len(inter_outs)
                lvl_bn_pfeats_p = [[]] * len(inter_outs)
                for bi, (b_feats_g, b_feats_p) in enumerate(
                    zip(bn_reid_feats_g, bn_reid_feats_p)
                ):
                    bn_num_lvls = [nums[bi] for nums in lvl_bn_nums]
                    bn_lvl_feats_g = torch.split(b_feats_g, bn_num_lvls)
                    bn_lvl_feats_p = torch.split(b_feats_p, bn_num_lvls)
                    for li in range(len(lvl_bn_pfeats_g)):
                        lvl_bn_pfeats_g[li].append(bn_lvl_feats_g[li])
                        lvl_bn_pfeats_p[li].append(bn_lvl_feats_p[li])
                aux_losses = {}
                for i in range(len(inter_outs)):
                    losses_g = self.compute_losses_lvl(
                        torch.cat(lvl_bn_pfeats_g[i], dim=0),
                        torch.cat(lvl_bn_ids[i], dim=0),
                        torch.cat(lvl_bn_asc[i], dim=0),
                        torch.cat(lvl_bn_logits[i], dim=0),
                        loss_layer_lvl=i,
                    )
                    losses_p = self.compute_losses_lvl_part(
                        torch.cat(lvl_bn_pfeats_p[i], dim=0),
                        torch.cat(lvl_bn_ids[i], dim=0),
                        torch.cat(lvl_bn_asc[i], dim=0),
                        torch.cat(lvl_bn_logits[i], dim=0),
                        loss_layer_lvl=i,
                    )
                    losses_p_n = {}
                    for k, v in losses_p.items():
                        losses_p_n[k + "_p"] = v
                    if i < len(inter_outs) - 1:
                        for k, v in losses_g.items():
                            aux_losses[k + "_{}".format(i)] = v
                        for k, v in losses_p.items():
                            aux_losses[k + "_{}".format(i)] = v
                    else:
                        head_outputs["losses"].update(losses_g)
                        head_outputs["losses"].update(losses_p)
                head_outputs["losses"].update(aux_losses)
                head_outputs["aux_outputs"] = head_aux_outputs

            elif self.shared_aux:
                # TODO metric feats and losses
                assert "aux_outputs" in det_outputs
                inter_outs = det_outputs["aux_outputs"] + [
                    {
                        "pred_logits": det_outputs["pred_logits"],
                        "pred_boxes": det_outputs["pred_boxes"],
                    }
                ]
                head_aux_outputs = []
                bs = inter_outs[0]["pred_boxes"].shape[0]
                bn_ids, bn_asc, bn_boxes, bn_logits = (
                    [[]] * bs,
                    [[]] * bs,
                    [[]] * bs,
                    [[]] * bs,
                )
                for i in range(len(inter_outs)):
                    (
                        assign_ids,
                        det_ids,
                        det_asc,
                        det_boxes,
                        det_logits,
                    ) = self.split_id_asc_box_logits(
                        inter_outs[i], targets, det_match_indices[i]
                    )
                    for bi in range(bs):
                        bn_ids[bi].append(det_ids[bi])
                        bn_asc[bi].append(det_asc[bi])
                        bn_boxes[bi].append(det_boxes[bi])
                        bn_logits.append(det_logits[bi])
                    if i < len(inter_outs) - 1:
                        head_aux_outputs.append({"assign_ids": assign_ids})
                    else:
                        head_outputs["assign_ids"] = assign_ids
                for bi in range(bs):
                    bn_ids[bi] = torch.cat(bn_ids[bi], dim=0)
                    bn_asc[bi] = torch.cat(bn_asc[bi], dim=0)
                    bn_boxes[bi] = torch.cat(bn_boxes[bi], dim=0)
                    bn_logits[bi] = torch.cat(bn_logits[bi], dim=0)
                if self.append_gt:
                    gt_ids, gt_asc, gt_boxes, gt_logits = self.get_gt_id_asc_box_logits(
                        targets
                    )
                    bs = assign_ids.shape[0]
                    for bi in range(bs):
                        bn_ids[bi] = torch.cat([bn_ids[bi], gt_ids[bi]], dim=0)
                        bn_asc[bi] = torch.cat([bn_asc[bi], gt_asc[bi]], dim=0)
                        bn_boxes[bi] = torch.cat([bn_boxes[bi], gt_boxes[bi]], dim=0)
                        bn_logits[bi] = torch.cat([bn_logits[bi], gt_logits[bi]], dim=0)
                bn_reid_feats_g = self.get_pfeats(bk_feats, bn_boxes)
                losses_g = self.compute_losses_lvl(
                    torch.cat(bn_reid_feats_g, dim=0),
                    torch.cat(bn_ids, dim=0),
                    torch.cat(bn_asc, dim=0),
                    torch.cat(bn_logits, dim=0),
                )
                bn_reid_feats_p, bn_part_assign = self.get_pfeats_part()
                losses_p = self.compute_losses_lvl_part(
                    torch.cat(bn_reid_feats_p, dim=0),
                    torch.cat(bn_ids, dim=0),
                    torch.cat(bn_asc, dim=0),
                    torch.cat(bn_logits, dim=0),
                    torch.cat(bn_part_assign, dim=0),
                )
                losses_p_n = {}
                for k, v in losses_p.items():
                    losses_p_n[k + "_p"] = v
                head_outputs["losses"].update(losses_g)
                head_outputs["losses"].update(losses_p_n)
                head_outputs["aux_outputs"] = head_aux_outputs
            else:
                # global
                (
                    assign_ids,
                    bn_ids,
                    bn_asc,
                    bn_boxes,
                    bn_logits,
                ) = self.split_id_asc_box_logits(
                    det_outputs, targets, det_match_indices[-1]
                )
                # compute only with valid ids
                if self.append_gt:
                    gt_ids, gt_asc, gt_boxes, gt_logits = self.get_gt_id_asc_box_logits(
                        targets
                    )
                    bs = assign_ids.shape[0]
                    for bi in range(bs):
                        bn_ids[bi] = torch.cat([bn_ids[bi], gt_ids[bi]], dim=0)
                        bn_asc[bi] = torch.cat([bn_asc[bi], gt_asc[bi]], dim=0)
                        bn_boxes[bi] = torch.cat([bn_boxes[bi], gt_boxes[bi]], dim=0)
                        bn_logits[bi] = torch.cat([bn_logits[bi], gt_logits[bi]], dim=0)
                bn_reid_feats_g_m, bn_reid_feats_g_c = self.get_pfeats(
                    bk_feats, bn_boxes
                )
                losses_g = self.compute_losses_lvl(
                    torch.cat(bn_reid_feats_g_c, dim=0),
                    torch.cat(bn_ids, dim=0),
                    torch.cat(bn_asc, dim=0),
                    torch.cat(bn_logits, dim=0),
                )
                if self.metric_loss is not None:
                    mlosses_g = self.compute_metric_losses(
                        torch.cat(bn_reid_feats_g_m, dim=0),
                        torch.cat(bn_ids, dim=0),
                        torch.cat(bn_asc, dim=0),
                        torch.cat(bn_logits, dim=0),
                    )
                    losses_g.update(mlosses_g)
                # part
                (
                    bn_reid_feats_p_m,
                    bn_reid_feats_p_c,
                    bn_part_assign,
                ) = self.get_pfeats_part()
                losses_p = self.compute_losses_lvl_part(
                    torch.cat(bn_reid_feats_p_c, dim=0),
                    torch.cat(bn_ids, dim=0),
                    torch.cat(bn_asc, dim=0),
                    torch.cat(bn_logits, dim=0),
                    torch.cat(bn_part_assign, dim=0),
                )
                if self.metric_loss is not None:
                    mlosses_p = self.compute_metric_losses(
                        torch.cat(bn_reid_feats_p_m, dim=0),
                        torch.cat(bn_ids, dim=0),
                        torch.cat(bn_asc, dim=0),
                        torch.cat(bn_logits, dim=0),
                        post_fix="p",
                    )
                    losses_p.update(mlosses_p)
                losses_p_n = {}
                for k, v in losses_p.items():
                    losses_p_n[k + "_p"] = v
                head_outputs["assign_ids"] = assign_ids
                head_outputs["losses"].update(losses_g)
                head_outputs["losses"].update(losses_p_n)
            if get_event_storage().iter % self.vis_period == 0:
                self.visualize_assign(targets, bn_boxes, bn_part_assign)
        else:
            if targets is None:
                # gallery feat
                det_boxes = det_outputs["pred_boxes"]
                _, det_pfeats_g = self.get_pfeats(bk_feats, det_boxes)
                _, det_pfeats_p, _ = self.get_pfeats_part()
                head_outputs["reid_feats"] = self.feats_fuse(det_pfeats_g, det_pfeats_p)
            else:
                # query feat
                _, _, bn_boxes, _ = self.get_gt_id_asc_box_logits(targets)
                _, pfeats_g = self.get_pfeats(bk_feats, bn_boxes)
                _, pfeats_p, bn_part_assign = self.get_pfeats_part()
                if self.vis_inf:
                    self.visualize_assign(
                        targets,
                        bn_boxes,
                        bn_part_assign,
                        save=os.path.join(self.vis_inf_dir, str(time.time()) + ".png"),
                    )
                head_outputs["reid_feats"] = self.feats_fuse(pfeats_g, pfeats_p)
        return head_outputs

    def get_roi_feats(self, bk_feats, d2_boxes):
        return self.box_pooler(bk_feats, d2_boxes)

    @torch.no_grad()
    def visualize_assign(self, targets, bn_boxes, bn_part_assign, save=""):
        # NOTE better way to get mean and std
        img_norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img_norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        trans_t2img_rgb_t = lambda t: (t * img_norm_std + img_norm_mean) * 255.0
        if self.training:
            storage = get_event_storage()
        tg_size = [384, 192]
        bs = len(bn_boxes)
        for bi in range(bs):
            boxes_bi = bn_boxes[bi].cpu()  # n x 4
            box_areas = (boxes_bi[:, 2] - boxes_bi[:, 0]) * boxes_bi[:, 3] - boxes_bi[
                :, 1
            ]
            sort_idxs = torch.argsort(box_areas, dim=0, descending=True)
            assigns_bi = bn_part_assign[bi].cpu()  # n x num_part x roi_h x roi_w
            idxs = sort_idxs[: min(10, sort_idxs.shape[0])]
            img_rgb_t = trans_t2img_rgb_t(targets[bi]["image_t"].cpu())  # 3 x h x w
            assigns_on_boxes = []
            for i in idxs:
                assign_on_box = _render_assign_on_box(
                    img_rgb_t, boxes_bi[i], assigns_bi[i], tgt_size=tg_size, save=save
                )
                assigns_on_boxes.append(assign_on_box)
            cat_assigns_on_boxes = torch.cat(assigns_on_boxes, dim=2)
            if self.training:
                storage.put_image("img_{}/seg".format(bi), cat_assigns_on_boxes / 255.0)

    def compute_losses_lvl_part(
        self,
        pfeats,
        assigned_ids,
        assign_scores,
        det_logits,
        part_assign_maps,
        loss_layer_lvl=-1,
    ):
        if "oim" in self.head_cfg.LOSS.NAME:
            oim_cfg = self.head_cfg.LOSS.OIM
            if oim_cfg.ADA_MM == "assign":
                lb_mms = (1 - assign_scores).view(-1) * oim_cfg.MM_FACTOR
            elif oim_cfg.ADA_MM == "obj":
                lb_mms = (1 - det_logits.detach().sigmoid()).view(
                    -1
                ) * oim_cfg.MM_FACTOR
            else:
                lb_mms = None
            losses = self.part_loss_layers[loss_layer_lvl](
                pfeats.view(-1, pfeats.shape[-1]), assigned_ids.view(-1), lb_mms
            )
        else:
            losses = self.part_loss_layers[loss_layer_lvl](pfeats, assigned_ids)
        # nonlap
        nonlap_loss = self.sl_loss_weight * shapingloss(
            part_assign_maps,
            self.sl_radius,
            self.sl_std,
            self.num_parts,
            self.sl_alpha,
            self.sl_beta,
            self.sl_eps,
        )
        losses["loss_nonlap"] = nonlap_loss
        return losses


class GroupingUnit(nn.Module):
    def __init__(self, in_channels, num_parts):
        super(GroupingUnit, self).__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels

        # params
        self.weight = nn.Parameter(
            torch.FloatTensor(num_parts, in_channels, 1, 1)
        )  # n * 1024 * 1*1
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))

    def reset_parameters(self, init_weight=None, init_smooth_factor=None):
        if init_weight is None:
            # msra init
            nn.init.kaiming_normal_(self.weight)
            self.weight.data.clamp_(min=1e-5)
        else:
            # init weight based on clustering
            assert init_weight.shape == (self.num_parts, self.in_channels)
            with torch.no_grad():
                self.weight.copy_(init_weight.unsqueeze(2).unsqueeze(3))

        # set smooth factor to 0 (before sigmoid)
        if init_smooth_factor is None:
            nn.init.constant_(self.smooth_factor, 0)
        else:
            # init smooth factor based on clustering
            assert init_smooth_factor.shape == (self.num_parts,)
            with torch.no_grad():
                self.smooth_factor.copy_(init_smooth_factor)

    def forward(self, inputs):
        assert inputs.dim() == 4

        # 0. store input size
        batch_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels == self.in_channels

        # 1. generate the grouping centers  # 5 1024 1 1 --> 1 5 1024 --> B 5 1024  # 因为
        grouping_centers = (
            self.weight.contiguous()
            .view(1, self.num_parts, self.in_channels)
            .expand(batch_size, self.num_parts, self.in_channels)
        )

        # 2. compute assignment matrix
        # - d = -\|X - C\|_2 = - X^2 - C^2 + 2 * C^T X
        # C^T X (N * K * H * W)
        inputs_cx = inputs.contiguous().view(
            batch_size, self.in_channels, input_h * input_w
        )
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        # X^2 (N * C * H * W) -> (N * 1 * H * W) -> (N * K * H * W)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1)
        # C^2 (K * C * 1 * 1) -> 1 * K * 1 * 1
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3)
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        # expand the smooth term
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(batch_size, -1, input_h, input_w)
        # assignment = softmax(-d/s) (-d must be negative)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1)  # default dim = 1

        # 3. compute residual coding
        # NCHW -> N * C * HW
        x = inputs.contiguous().view(batch_size, self.in_channels, -1)
        # permute the inputs -> N * HW * C
        x = x.permute(0, 2, 1)

        # compute weighted feats N * K * C
        assign = assign.contiguous().view(batch_size, self.num_parts, -1)
        qx = torch.bmm(assign, x)

        # repeat the graph_weights (K * C) -> (N * K * C)
        c = grouping_centers

        # sum of assignment (N * K * 1) -> (N * K * K)
        sum_ass = torch.sum(assign, dim=2, keepdim=True)

        # residual coding N * K * C
        sum_ass = sum_ass.expand(-1, -1, self.in_channels).clamp(min=1e-5)
        sigma = (beta / 2).sqrt()
        out = ((qx / sum_ass) - c) / sigma.unsqueeze(0).unsqueeze(2)

        # 4. prepare outputs
        # we need to memorize the assignment (N * K * H * W)
        assign = assign.contiguous().view(batch_size, self.num_parts, input_h, input_w)

        # output features has the size of N * K * C
        outputs = nn.functional.normalize(out, dim=2)  # b 5 1024
        outputs_t = outputs.permute(0, 2, 1)  # b 1024 5

        # generate assignment map for basis for visualization
        return outputs_t, assign

    # name
    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_channels)
            + " -> "
            + str(self.num_parts)
            + ")"
        )


class BatchNorm(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        weight_freeze=False,
        bias_freeze=False,
        weight_init=1.0,
        bias_init=0.0,
        **kwargs,
    ):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None:
            nn.init.constant_(self.weight, weight_init)
        if bias_init is not None:
            nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def _render_assign_on_box(
    img_rgb_t, pbox_t: Tensor, assign_box, tgt_size=(384, 192), save=""
):

    # NOTE split two view for clearity
    pbox_int = pbox_t.int()  # xyxy
    pbox_img_rgb_t = img_rgb_t[
        :,
        max(pbox_int[1], 0) : min(pbox_int[3] + 1, img_rgb_t.shape[1]),
        max(pbox_int[0], 0) : min(pbox_int[2] + 1, img_rgb_t.shape[2]),
    ].clone()  # 3 x h x w
    pbox_img_rgb_t = torch.nn.functional.interpolate(
        pbox_img_rgb_t[None], size=tgt_size, mode="bilinear", align_corners=False
    ).squeeze(0)

    assign_reshaped = torch.nn.functional.interpolate(
        assign_box[None], size=tgt_size, mode="bilinear", align_corners=False
    ).squeeze(0)
    n_parts = assign_reshaped.shape[0]
    _, assign = torch.max(assign_reshaped, dim=0)
    assign = assign.cpu()
    # generate colors
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / n_parts):
        hue = i / 360.0
        lightness = 0.5
        saturation = 0.9
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    colors_np = torch.tensor(colors, dtype=torch.float32) * 255.0
    # merge
    coeff = 0.3
    for i in range(assign.shape[0]):
        for j in range(assign.shape[1]):
            assign_ij = assign[i][j]
            pbox_img_rgb_t[:, i, j] = (1 - coeff) * pbox_img_rgb_t[
                :, i, j
            ] + coeff * colors_np[assign_ij]
    if len(save) > 0:
        img_arr = pbox_img_rgb_t.permute(1, 2, 0).numpy().astype(np.uint)
        cv2.imwrite(save, img_arr)
    return pbox_img_rgb_t
    """assign_map = pbox_img_rgb_t.clone()
    for i in range(assign.shape[0]):
        for j in range(assign.shape[1]):
            assign_map[:, i, j] = colors_np[assign[i][j]]
    return add_map(pbox_img_rgb_t,assign_map)#torch.cat([pbox_img_rgb_t, assign_map], dim=2)"""


def add_map(pimg_t, map_t):
    import cv2

    pimg_arr = pimg_t.permute(1, 2, 0).numpy()
    map_arr = map_t.permute(1, 2, 0).numpy()
    add_img = cv2.addWeighted(src1=pimg_arr, alpha=0.6, src2=map_arr, beta=0.4, gamma=0)
    return torch.tensor(add_img).permute(2, 0, 1)
