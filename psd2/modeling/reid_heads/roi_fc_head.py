from typing import Iterable
import torch
import os
import time
from .base_reid_head import ReidHeadBase
from psd2.modeling.poolers import ROIPooler
from psd2.structures import Boxes
import torch.nn as nn
from torch.nn import init
import numpy as np
import cv2
from psd2.utils.events import get_event_storage
from psd2.layers.pooling import *
import torch.nn.functional as tF


class RoiFcHead(ReidHeadBase):
    def __init__(self, head_cfg):
        super().__init__(head_cfg)
        pooler_cfg = head_cfg.ROI_POOLER
        self.train_feat_at = head_cfg.PERSON_FEATURE.TRAIN_AT
        self.inf_feat_at = head_cfg.PERSON_FEATURE.INF_AT
        self._init_modules(
            roi_size=pooler_cfg.POOLER_RESOLUTION,
            roi_scales=pooler_cfg.POOLER_SCALES,
            samp_ratio=pooler_cfg.POOLER_SAMPLING_RATIO,
            roipooler_type=pooler_cfg.POOLER_TYPE,
            pool_layer=head_cfg.POOL_LAYER,
        )
        # visualization
        self.vis_period = head_cfg.VIS_PERIOD
        self.vis_inf = head_cfg.VIS_INF
        self.vis_inf_dir = head_cfg.VIS_INF_SAVE
        if self.vis_inf and not os.path.exists(self.vis_inf_dir):
            os.makedirs(self.vis_inf_dir)

    def _init_modules(
        self, roi_size, roi_scales, samp_ratio, roipooler_type, pool_layer
    ):
        self.box_pooler = ROIPooler(
            output_size=roi_size,
            scales=roi_scales,
            sampling_ratio=samp_ratio,
            pooler_type=roipooler_type,
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

    def get_pfeats(self, bk_feats, roi_boxes):
        if isinstance(roi_boxes, torch.Tensor):
            bs, qs = roi_boxes.shape[:2]
            boxes_list = [Boxes(roi_boxes[bi]) for bi in range(bs)]
            rois_feats = self.get_roi_feats(
                bk_feats, boxes_list
            )  # (B x Nq) x 256 x 24 x 12
            mpfeats, cpfeats = self.pfeat_head(rois_feats)
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
            rois_feats = self.get_roi_feats(
                bk_feats, boxes_list
            )  # ( b0 + b1+ ...) x 256 x 24 x 8
            mpfeats, cpfeats = self.pfeat_head(rois_feats)
            return (
                torch.split(mpfeats, num_splits),
                torch.split(cpfeats, num_splits),
                torch.split(rois_feats, num_splits),
            )

    def pfeat_head(self, rois_feats):
        proj_feats = self.feat_proj(rois_feats)
        if proj_feats.dim() > 2:
            proj_feats = proj_feats.reshape(proj_feats.shape[0], proj_feats.shape[1])
        after_bn_feat = self.feat_bn(proj_feats)
        if self.training:
            feat_at = self.train_feat_at
        else:
            feat_at = self.inf_feat_at
        if feat_at == "after_bn":
            cls_feats = after_bn_feat
        elif feat_at == "before_bn":
            cls_feats = proj_feats
        else:
            raise KeyError(f"{feat_at} is not supported!")
        if self.metric_feat_at == "after_bn":
            metric_feats = after_bn_feat
        elif self.metric_feat_at == "before_bn":
            metric_feats = proj_feats
        else:
            raise KeyError(f"{self.metric_feat_at} is not supported!")
        return metric_feats, cls_feats

    def get_roi_feats(self, bk_feats, d2_boxes):
        return self.box_pooler(bk_feats, d2_boxes)

    @torch.no_grad()
    def visualize_(self, targets, bn_boxes, bn_roi_feats, save=""):
        # NOTE better way to get mean and std
        img_norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img_norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        trans_t2img_rgb_t = lambda t: (t * img_norm_std + img_norm_mean) * 255.0
        if self.training:
            storage = get_event_storage()
        tg_size = [384, 128]
        bs = len(bn_boxes)
        for bi in range(bs):
            boxes_bi = bn_boxes[bi].cpu()  # n x 4
            box_areas = (boxes_bi[:, 2] - boxes_bi[:, 0]) * boxes_bi[:, 3] - boxes_bi[
                :, 1
            ]
            sort_idxs = torch.argsort(box_areas, dim=0, descending=True)
            feats_bi = bn_roi_feats[bi].cpu()  # n x num_part x roi_h x roi_w
            idxs = sort_idxs[: min(10, sort_idxs.shape[0])]
            img_rgb_t = trans_t2img_rgb_t(targets[bi]["image_t"].cpu())  # 3 x h x w
            assigns_on_boxes = []
            for i in idxs:
                assign_on_box = _render_attn_on_box(
                    img_rgb_t, boxes_bi[i], feats_bi[i], tgt_size=tg_size, save=save
                )
                assigns_on_boxes.append(assign_on_box)
            cat_assigns_on_boxes = torch.cat(assigns_on_boxes, dim=2)
            if self.training:
                storage.put_image(
                    "img_{}/attn".format(bi), cat_assigns_on_boxes / 255.0
                )

    def forward(self, bk_feats, det_outputs, targets, det_match_indices, *args, **kw):
        head_outputs = {}
        if self.training:
            # NOTE update to only involve boxes with valid ids
            head_outputs["losses"] = {}
            if self.append_gt:
                assert targets is not None
            if len(self.loss_layers) > 1:
                # TODO compute only with ids
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
                bn_reid_feats_metric, bn_reid_feats_cls, bn_roi_feats = self.get_pfeats(
                    bk_feats, bn_boxes
                )  # bn x lvl
                lvl_bn_pfeats_metric = [[]] * len(inter_outs)
                lvl_bn_pfeats_cls = [[]] * len(inter_outs)
                for bi, (b_feats_metric, b_feats_cls) in enumerate(
                    zip(bn_reid_feats_metric, bn_reid_feats_cls)
                ):
                    bn_num_lvls = [nums[bi] for nums in lvl_bn_nums]
                    bn_lvl_feats_metric = torch.split(b_feats_metric, bn_num_lvls)
                    bn_lvl_feats_cls = torch.split(b_feats_cls, bn_num_lvls)
                    for li in range(len(lvl_bn_pfeats_cls)):
                        lvl_bn_pfeats_cls[li].append(bn_lvl_feats_cls[li])
                        lvl_bn_pfeats_metric[li].append(bn_lvl_feats_metric[li])
                aux_losses = {}
                for i in range(len(inter_outs)):
                    losses = self.compute_losses_lvl(
                        torch.cat(lvl_bn_pfeats_cls[i], dim=0),
                        torch.cat(lvl_bn_ids[i], dim=0),
                        torch.cat(lvl_bn_asc[i], dim=0),
                        torch.cat(lvl_bn_logits[i], dim=0),
                        loss_layer_lvl=i,
                    )
                    if self.metric_loss is not None:
                        metric_losses = self.compute_metric_losses(
                            torch.cat(lvl_bn_pfeats_metric[i], dim=0),
                            torch.cat(lvl_bn_ids[i], dim=0),
                            torch.cat(lvl_bn_asc[i], dim=0),
                            torch.cat(lvl_bn_logits[i], dim=0),
                        )
                        losses.update(metric_losses)
                    if i < len(inter_outs) - 1:
                        for k, v in losses.items():
                            aux_losses[k + "_{}".format(i)] = v
                    else:
                        head_outputs["losses"].update(losses)
                head_outputs["losses"].update(aux_losses)
                head_outputs["aux_outputs"] = head_aux_outputs

            elif self.shared_aux:
                # TODO compute only with ids
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
                bn_reid_feats_metric, bn_reid_feats_cls, bn_roi_feats = self.get_pfeats(
                    bk_feats, bn_boxes
                )
                losses = self.compute_losses_lvl(
                    torch.cat(bn_reid_feats_cls, dim=0),
                    torch.cat(bn_ids, dim=0),
                    torch.cat(bn_asc, dim=0),
                    torch.cat(bn_logits, dim=0),
                )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        torch.cat(bn_reid_feats_metric, dim=0),
                        torch.cat(bn_ids, dim=0),
                        torch.cat(bn_asc, dim=0),
                        torch.cat(bn_logits, dim=0),
                    )
                    losses.update(metric_losses)
                head_outputs["losses"].update(losses)
                head_outputs["aux_outputs"] = head_aux_outputs
            else:
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
                bn_reid_feats_metric, bn_reid_feats_cls, bn_roi_feats = self.get_pfeats(
                    bk_feats, bn_boxes
                )
                losses = self.compute_losses_lvl(
                    torch.cat(bn_reid_feats_cls, dim=0),
                    torch.cat(bn_ids, dim=0),
                    torch.cat(bn_asc, dim=0),
                    torch.cat(bn_logits, dim=0),
                )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        torch.cat(bn_reid_feats_metric, dim=0),
                        torch.cat(bn_ids, dim=0),
                        torch.cat(bn_asc, dim=0),
                        torch.cat(bn_logits, dim=0),
                    )
                    losses.update(metric_losses)
                head_outputs["assign_ids"] = assign_ids
                # head_outputs["reid_feats"] = reid_feats
                head_outputs["losses"].update(losses)
            if get_event_storage().iter % self.vis_period == 0:
                self.visualize_(targets, bn_boxes, bn_roi_feats)
        else:
            if targets is None:
                # gallery feat
                det_boxes = det_outputs["pred_boxes"]
                _, det_pfeats, bn_roi_feats = self.get_pfeats(bk_feats, det_boxes)
                head_outputs["reid_feats"] = tF.normalize(det_pfeats, dim=-1)
            else:
                # query feat
                _, _, bn_boxes, _ = self.get_gt_id_asc_box_logits(targets)
                _, pfeats, bn_roi_feats = self.get_pfeats(bk_feats, bn_boxes)
                norm_feats = []
                for pfeat in pfeats:
                    norm_feats.append(tF.normalize(pfeat, dim=-1))
                head_outputs["reid_feats"] = norm_feats
                if self.vis_inf:
                    self.visualize_(
                        targets,
                        bn_boxes,
                        bn_roi_feats,
                        save=os.path.join(self.vis_inf_dir, str(time.time()) + ".png"),
                    )
        return head_outputs


def _render_attn_on_box(img_rgb_t, pbox_t, attn_box, tgt_size=(384, 192), save=""):
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

    attn_reshaped = torch.nn.functional.interpolate(
        attn_box[None], size=tgt_size, mode="bilinear", align_corners=False
    ).squeeze(0)
    attn_reshaped = (attn_reshaped ** 2).sum(0)
    attn_reshaped = (
        255
        * (attn_reshaped - attn_reshaped.min())
        / (attn_reshaped.max() - attn_reshaped.min() + 1e-12)
    )
    attn_img = attn_reshaped.cpu().numpy().astype(np.uint8)
    attn_img = cv2.applyColorMap(attn_img, cv2.COLORMAP_JET)  # bgr hwc
    attn_img = torch.tensor(
        attn_img[..., ::-1].copy(), device=img_rgb_t.device, dtype=img_rgb_t.dtype
    ).permute(2, 0, 1)
    coeff = 0.3
    pbox_img_rgb_t = (1 - coeff) * pbox_img_rgb_t + coeff * attn_img
    if len(save) > 0:
        img_arr = pbox_img_rgb_t.permute(1, 2, 0).numpy().astype(np.uint)
        cv2.imwrite(save, img_arr)
    return pbox_img_rgb_t


class RoiFcHeadSide(RoiFcHead):
    def _init_modules(
        self, roi_size, roi_scales, samp_ratio, roipooler_type, pool_layer
    ):
        self.box_pooler = ROIPooler(
            output_size=roi_size,
            scales=roi_scales,
            sampling_ratio=samp_ratio,
            pooler_type=roipooler_type,
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

    def pfeat_head(self, rois_feats):
        alpha_weight = torch.sigmoid(self.side_alpha)
        proj_feats = self.feat_proj(rois_feats)
        proj_feats_side = self.feat_proj_side(rois_feats)
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
