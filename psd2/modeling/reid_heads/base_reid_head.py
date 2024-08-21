import torch.nn as nn
import torch
import functools
import torchvision
from psd2.layers.mem_matching_losses import build_loss_layer
import copy
from psd2.layers.metric_loss import TripletLoss
from psd2.utils.events import get_event_storage
import psd2.utils.comm as comm
import torch.nn.functional as tF
from psd2.structures.boxes import box_xyxy_to_cxcywh
import os

INF = 1e10
EPS = 1e-10

"""
det_outputs:
    {
        "pred_boxes": B x N x 4,
        "pred_logits": B x N,
        "aux_outputs" (optional): 
            [
                {
                    "pred_boxes": B x N x 4,
                    "pred_logits": B x N,
                },
                ...
            ]
    }
targets:
    [
        {
            "file_name": fname,
            "image_id": imgid,
            "boxes": bboxes,
            "ids": ids,
            ...(optinal)
        },
        ...
    ]
    
"""


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ReidHeadBase(nn.Module):
    def __init__(self, head_cfg):
        super().__init__()
        self.head_cfg = head_cfg
        if hasattr(head_cfg.ID_ASSIGN, "POS_IOU_THRED"):
            self.pos_iou_thred = head_cfg.ID_ASSIGN.POS_IOU_THRED
        if hasattr(head_cfg.ID_ASSIGN, "POS_CENTER_RATIO"):
            self.pos_center_ratio = head_cfg.ID_ASSIGN.POS_CENTER_RATIO
        self.pos_score_thred = head_cfg.ID_ASSIGN.POS_SCORE_THRED
        self._id_assign_method = self._get_id_assign_method(head_cfg.ID_ASSIGN.NAME)
        self.pfeat_dim = head_cfg.PERSON_FEATURE.DIM
        self.append_gt = head_cfg.APPEND_GT
        loss_cfg = head_cfg.LOSS
        loss_layer = build_loss_layer(loss_cfg, head_cfg.PERSON_FEATURE.DIM)
        self.out_lvls = loss_cfg.AUX_LOSS + 1
        if loss_cfg.AUX_LOSS > 0:
            self.loss_layers = _get_clones(loss_layer, loss_cfg.AUX_LOSS)
            self.loss_layers.append(loss_layer)
        else:
            self.loss_layers = nn.ModuleList([loss_layer])

        self.shared_aux = loss_cfg.SHARED_AUX
        self.in_channels = head_cfg.IN_CHANNELS
        self.box_gradient = head_cfg.BOX_BP
        metric_loss_cfg = head_cfg.LOSS.METRIC
        if metric_loss_cfg.NAME == "triplet":
            self.metric_loss = TripletLoss(metric_loss_cfg.MARGIN, reduction="mean")
            self.norm_triplet = metric_loss_cfg.NORM_FEAT
        else:
            self.metric_loss = None
        self.metric_feat_at = metric_loss_cfg.FEAT_AT

        # visualization
        self.vis_period = head_cfg.VIS_PERIOD
        self.vis_inf = head_cfg.VIS_INF
        self.vis_inf_dir = head_cfg.VIS_INF_SAVE
        if self.vis_inf and not os.path.exists(self.vis_inf_dir):
            os.makedirs(self.vis_inf_dir)

    def _get_id_assign_method(self, assign_type):
        return {
            "set": self._assign_id_via_set,
            "iou": self._assign_id_via_iou,
            "adaiou": functools.partial(self._assign_id_via_iou, adamm=True),
            "ciou": self._assign_id_via_ctnsciou,
            "adaset": functools.partial(self._assign_id_via_set, adamm=True),
            "siou": self._assign_id_via_iou,
            "fovea_center": self._assign_id_via_fvcenter,
        }[assign_type]

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @torch.no_grad()
    def _assign_id_via_set(self, det_outputs, targets, indices):
        """
        id>=0 labeled person
        id=-1 unlabeled person
        id=-2 background
        """
        bs, ns = det_outputs["pred_boxes"].shape[:2]
        output_ids = (
            torch.zeros(
                bs, ns, dtype=torch.long, device=det_outputs["pred_boxes"].device
            )
            - 2
        )
        b_idx, s_idx = self._get_src_permutation_idx(indices)
        t_idx = torch.cat([t for (_, t) in indices])
        gt_ids = []
        for bid, tid in zip(b_idx, t_idx):
            gt_ids.append(targets[bid]["ids"][tid])
        gt_ids = output_ids.new_tensor(gt_ids)
        output_ids[(b_idx, s_idx)] = gt_ids
        # ious as  momentums

        # mms = torch.zeros_like(output_ids, dtype=torch.float32) + 1
        """
        ious_bs = []
        for bi in range(bs):
            gt_bboxes = targets[bi]["boxes"]  # ng x 4, cx-cy-w-h
            pred_bboxes = det_outputs["pred_boxes"][bi]  # np x 4, cx-cy-w-h
            pairwise_ious = torchvision.ops.box_iou(pred_bboxes, gt_bboxes)  # np x ng
            pairwise_ious[pairwise_ious < self.pos_iou_thred] = 0
            ious_bs.append(pairwise_ious)
        out_scores = det_outputs["pred_logits"].squeeze(2).sigmoid()
        output_ids[out_scores < self.pos_score_thred] = -2
        """
        """match_mms = []
        for bid, sid, tid in zip(b_idx, s_idx, t_idx):
            match_mms.append(1 - ious_bs[bid][sid, tid])
        match_mms = mms.new_tensor(match_mms)
        mms[(b_idx, s_idx)] = match_mms"""
        return (
            output_ids,
            None,
        )  # NOTE disabled for now, torch.cat(ious_bs, dim=0)  # , mms

    @torch.no_grad()
    def _assign_id_via_ctnsciou(self, det_outputs, targets, indices):
        # center-sampling like, plus iou constraint
        center_factor = 0.5
        bs, ns = det_outputs["pred_boxes"].shape[:2]
        output_ids = (
            torch.zeros(
                bs, ns, dtype=torch.long, device=det_outputs["pred_boxes"].device
            )
            - 2
        )
        # mms = torch.zeros_like(output_ids, dtype=torch.float32) + 1
        ref_pts = det_outputs["pred_ref_pts"]  # b x n x 2
        n_ponits = ref_pts.shape[1]
        for bi, img_target in enumerate(targets):
            gt_bboxes = img_target["boxes"]  # ng x 4, cx-cy-w-h
            pred_bboxes = det_outputs["pred_boxes"][bi]  # np x 4, cx-cy-w-h
            pairwise_ious = torchvision.ops.box_iou(pred_bboxes, gt_bboxes)  # np x ng
            n_gts = gt_bboxes.shape[0]
            bi_ref_pts = ref_pts[bi]  # np x 2
            expd_gt_bboxes = gt_bboxes[None].expand(n_ponits, n_gts, 4)  # np x ng x4
            expd_ref_pts = bi_ref_pts.unsqueeze(1).expand(n_ponits, n_gts, 2)
            # inside a `center bbox`
            radius = (
                expd_gt_bboxes[..., 2:] / 2 * center_factor
            )  # np x ng x 2, 0.5 is shrunk factor
            x_mins = expd_gt_bboxes[..., 0] - radius[..., 0]  # np x ng
            y_mins = expd_gt_bboxes[..., 1] - radius[..., 1]  # np x ng
            x_maxs = expd_gt_bboxes[..., 0] + radius[..., 0]  # np x ng
            y_maxs = expd_gt_bboxes[..., 1] + radius[..., 1]  # np x ng

            pairwise_dists = torch.cdist(
                bi_ref_pts[None], gt_bboxes[:, :2][None]
            ).squeeze(
                0
            )  # np x ng
            around_center_masks = [
                expd_ref_pts[..., 0] >= x_mins,
                expd_ref_pts[..., 1] >= y_mins,
                expd_ref_pts[..., 0] <= x_maxs,
                expd_ref_pts[..., 1] <= y_maxs,
            ]
            in_center_box_mask = functools.reduce(
                torch.logical_and, around_center_masks
            )
            pairwise_dists[in_center_box_mask == 0] = INF
            pairwise_dists[pairwise_ious < self.pos_iou_thred] = INF
            min_dist, min_dist_inds = pairwise_dists.min(dim=1)
            output_ids[bi] = min_dist_inds.new_tensor(img_target["ids"])[min_dist_inds]
            output_ids[bi][min_dist == INF] = -2

        # centerness dist
        rx = bi_ref_pts[..., 0].view(-1, 1)  # bn x 1
        ry = bi_ref_pts[..., 1].view(-1, 1)  # bn x 1
        xc = gt_bboxes[:, 0].unsqueeze(0)  # 1 x g
        yc = gt_bboxes[:, 1].unsqueeze(0)  # 1 x g
        cbw = gt_bboxes[:, 2].unsqueeze(0) * center_factor  # 1 x g
        cbh = gt_bboxes[:, 3].unsqueeze(0) * center_factor  # 1 x g
        l_ct = rx - (xc - cbw / 2)  # bn x g
        r_ct = (xc + cbw / 2) - rx  # bn x g
        t_ct = (yc + cbh / 2) - ry  # bn x g
        b_ct = ry - (yc - cbh / 2)  # bn x g
        ct_sq = (
            torch.where(l_ct < r_ct, l_ct, r_ct)
            / torch.where(l_ct > r_ct, l_ct, r_ct)
            * torch.where(t_ct < b_ct, t_ct, b_ct)
            / torch.where(t_ct > b_ct, t_ct, b_ct)
        )
        valid_masks = [l_ct <= 0, r_ct <= 0, t_ct <= 0, b_ct <= 0]
        out_box_mask = functools.reduce(torch.logical_or, valid_masks)
        ct_sq[out_box_mask] = 0
        pairwise_ctns = ct_sq ** (0.5)
        rids = min_dist_inds.new_tensor(list(range(min_dist_inds.shape[0])))
        ctns = pairwise_ctns[(rids, min_dist_inds)]
        # iou metric
        # pairwise_ious[pairwise_ious<iou_thred]=0

        corres_ious = pairwise_ious[(rids, min_dist_inds)]
        ctns_ciou = ctns * corres_ious
        return output_ids, ctns_ciou

    @torch.no_grad()
    def _assign_id_via_iou(self, det_outputs, targets, indices):
        # iou only
        bs, ns = det_outputs["pred_boxes"].shape[:2]
        output_ids = (
            torch.zeros(
                bs, ns, dtype=torch.long, device=det_outputs["pred_boxes"].device
            )
            - 2
        )
        bt_ious = []
        # mms = torch.zeros_like(output_ids, dtype=torch.float32) + 1
        for bi, img_target in enumerate(targets):
            gt_bboxes = img_target["boxes"]  # ng x 4, cx-cy-w-h
            pred_bboxes = det_outputs["pred_boxes"][bi]  # np x 4, x-y-w-h
            pairwise_ious = torchvision.ops.box_iou(pred_bboxes, gt_bboxes)  # np x ng
            pairwise_ious[pairwise_ious < self.pos_iou_thred] = 0
            max_iou, max_iou_inds = pairwise_ious.max(dim=1)
            output_ids[bi] = img_target["ids"][max_iou_inds]
            output_ids[bi][max_iou == 0] = -2
            bt_ious.append(max_iou)
            # mms[bi] = 1 - max_iou
        out_scores = det_outputs["pred_logits"].squeeze(2).sigmoid()
        output_ids[out_scores < self.pos_score_thred] = -2
        return output_ids, torch.stack(bt_ious, dim=0)  # , mms

    @torch.no_grad()
    def _assign_id_via_fvcenter(self, det_outputs, targets, indices):
        if "assign_ref_pts" in det_outputs:
            ref_pts = det_outputs["pred_ref_pts"].detach()
        elif "locations" in det_outputs:
            ref_pts = det_outputs["locations"].detach()
        else:
            pred_boxes = det_outputs["pred_boxes"]  # b x n x 4 xyxy_abs
            # aug_whwh_bs = torch.stack([v["aug_whwh"] for v in targets])
            # pred_boxes_ccwh_rel = box_xyxy_to_cxcywh(
            #    (pred_boxes / aug_whwh_bs).flatten(0, 1)
            # ).reshape(*pred_boxes.shape)
            pred_boxes_ccwh = box_xyxy_to_cxcywh(pred_boxes.flatten(0, 1)).reshape(
                *pred_boxes.shape
            )
            ref_pts = pred_boxes_ccwh[:, :, :2].detach()  # b x n x 2
        bs, ns = ref_pts.shape[:2]
        output_ids = (
            torch.zeros(
                bs, ns, dtype=torch.long, device=det_outputs["pred_boxes"].device
            )
            - 2
        )
        for bi, img_target in enumerate(targets):
            gt_bboxes = img_target["boxes"]  # ng x 4, xyxy_abs
            gt_bboxes = box_xyxy_to_cxcywh(gt_bboxes)
            # fovea region
            x_min = (
                gt_bboxes[:, 0] - 0.5 * self.pos_center_ratio * gt_bboxes[:, 2]
            )  # ng
            x_max = gt_bboxes[:, 0] + 0.5 * self.pos_center_ratio * gt_bboxes[:, 2]
            y_min = gt_bboxes[:, 1] - 0.5 * self.pos_center_ratio * gt_bboxes[:, 3]
            y_max = gt_bboxes[:, 1] + 0.5 * self.pos_center_ratio * gt_bboxes[:, 3]
            # in fovea mask
            bi_ref_pts = ref_pts[bi]  # n x 2
            in_mask = torch.logical_and(
                bi_ref_pts[:, None][..., 0] >= x_min[None],
                bi_ref_pts[:, None][..., 0] <= x_max[None],
            )  # n x ng
            in_mask = torch.logical_and(
                in_mask, bi_ref_pts[:, None][..., 1] >= y_min[None]
            )
            in_mask = torch.logical_and(
                in_mask, bi_ref_pts[:, None][..., 1] <= y_max[None]
            )
            box_areas = gt_bboxes[:, -1] * gt_bboxes[:, -2]
            in_mask = in_mask.int()
            in_box_nums = in_mask.sum(dim=1)  # qi
            amb_qi_mask = in_box_nums > 1
            amb_qi_idxs = torch.nonzero(amb_qi_mask).squeeze(1)
            for qi in amb_qi_idxs:
                mask_line = in_mask[qi]
                in_boxes_idxs = torch.nonzero(mask_line).squeeze(1)  # k
                areas = box_areas[in_boxes_idxs]
                min_area_idx = torch.argmin(areas)
                keep_idx = in_boxes_idxs[min_area_idx]
                mask_line[in_boxes_idxs] = 0
                mask_line[keep_idx] = 1
            mask_v, mask_inds = in_mask.max(dim=1)
            output_ids[bi] = img_target["ids"][mask_inds]
            output_ids[bi][mask_v == 0] = -2
        out_scores = det_outputs["pred_logits"].squeeze(2).sigmoid()
        output_ids[out_scores < self.pos_score_thred] = -2
        # TODO ctns mms
        return output_ids, None

    def get_pfeats(self, bk_feats, roi_boxes):
        raise NotImplementedError

    def split_id_asc_box_logits(self, det_outputs, targets, indices):
        ids, as_s = self._id_assign_method(det_outputs, targets, indices)
        det_ids = ids
        bn_ids = []
        bn_asc = []
        bn_boxes = []
        bn_logits = []
        bn = ids.shape[0]
        for bi in range(bn):
            mask = ids[bi] > -2
            bn_ids.append(ids[bi][mask])
            bn_asc.append(as_s[bi][mask])
            if self.box_gradient:
                bn_boxes.append(det_outputs["pred_boxes"][bi][mask])
            else:
                bn_boxes.append(det_outputs["pred_boxes"][bi][mask].detach())
            bn_logits.append(det_outputs["pred_logits"][bi][mask])
        return det_ids, bn_ids, bn_asc, bn_boxes, bn_logits

    def get_gt_id_asc_box_logits(self, targets):
        bn_ids = []
        bn_asc = []
        bn_boxes = []
        bn_logits = []
        for item in targets:
            np = item["ids"].shape[0]
            bn_ids.append(item["ids"])
            bn_asc.append(torch.ones(np, device=item["ids"].device))
            bn_boxes.append(item["boxes"])
            bn_logits.append(torch.zeros((np, 1), device=item["ids"].device) + INF)
        return bn_ids, bn_asc, bn_boxes, bn_logits

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
                bn_reid_feats_metric, bn_reid_feats_cls = self.get_pfeats(
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
                bn_reid_feats_metric, bn_reid_feats_cls = self.get_pfeats(
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
                bn_reid_feats_metric, bn_reid_feats_cls = self.get_pfeats(
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

        else:
            if targets is None:
                # gallery feat
                det_boxes = det_outputs["pred_boxes"]
                _, det_pfeats = self.get_pfeats(bk_feats, det_boxes)
                head_outputs["reid_feats"] = tF.normalize(det_pfeats, dim=-1)
            else:
                # query feat
                _, _, bn_boxes, _ = self.get_gt_id_asc_box_logits(targets)
                _, pfeats = self.get_pfeats(bk_feats, bn_boxes)
                norm_feats = []
                for pfeat in pfeats:
                    norm_feats.append(tF.normalize(pfeat, dim=-1))
                """n_pfeats = len(pfeats)
                cat_pfeats = torch.cat(pfeats, dim=0)
                norm_cat_feats = tF.normalize(cat_pfeats, dim=-1)
                head_outputs["reid_feats"] = torch.split(norm_cat_feats, n_pfeats)
                """
                head_outputs["reid_feats"] = norm_feats
        return head_outputs

    def compute_losses_lvl(
        self, pfeats, assigned_ids, assign_scores, det_logits, loss_layer_lvl=-1
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
            losses = self.loss_layers[loss_layer_lvl](
                pfeats.flatten(0, -2), assigned_ids.view(-1), lb_mms
            )
        else:
            losses = self.loss_layers[loss_layer_lvl](pfeats, assigned_ids)
        return losses

    def compute_metric_losses(
        self, pfeats, assigned_ids, assign_scores, det_logits, post_fix=""
    ):
        # NOTE triplet only for now
        feats = pfeats.view(-1, pfeats.shape[-1])
        pids = assigned_ids.view(-1)
        # sync data
        sync_data = (feats, pids)
        comm.synchronize()
        all_syncs = comm.all_gather(sync_data)
        all_feats = [feats]
        all_pids = [pids]
        cur_device_idx = feats.device.index
        for feats_n, pids_n in all_syncs:
            feats_device_idx = feats_n.device.index
            if feats_device_idx != cur_device_idx:
                # TODO check if correct to detach
                all_feats.append(feats_n.to(feats.device).detach())
                all_pids.append(pids_n.to(pids.device).detach())
        feats = torch.cat(all_feats, dim=0)
        pids = torch.cat(all_pids, dim=0)
        id_mask = pids > -1
        tp_losses = self.metric_loss(
            feats[id_mask],
            feats,
            pids[id_mask],
            pids,
            normalize_feature=self.norm_triplet,
        )
        losses = {}
        for k, v in tp_losses.items():
            if "loss" in k:
                losses[k] = v
            else:
                get_event_storage().put_scalar(k + post_fix, v)
        return losses
