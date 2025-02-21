import torch
import torch.nn as nn
from fdps.utils import comm
import fdps.structures.boxes as iou_tools
import copy
from torch import nn
import torch.nn.functional as F
import copy

# util functions


@torch.no_grad()
def _accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SrcnnSetCriterion(nn.Module):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        use_focal,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        if self.use_focal:
            self.focal_loss_alpha = focal_alpha
            self.focal_loss_gamma = focal_gamma
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        if self.use_focal:
            src_logits = src_logits.flatten(0, 1)
            # prepare one_hot target.
            target_classes = target_classes.flatten(0, 1)
            pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[
                0
            ]
            labels = torch.zeros_like(src_logits)
            labels[pos_inds, target_classes[pos_inds]] = 1
            # comp focal loss.
            class_loss = (
                sigmoid_focal_loss_jit(
                    src_logits,
                    labels,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                )
                / num_boxes
            )
            losses = {"loss_ce": class_loss}
        else:
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
            losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = (
                100 - _accuracy(src_logits[idx], target_classes_o)[0]
            )
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        aug_whwh_bs = torch.cat([v["aug_whwh"] for v in targets], dim=0)

        losses = {}
        loss_giou = 1 - torch.diag(
            iou_tools.generalized_box_iou(src_boxes, target_boxes)
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        aug_whwh_bs = aug_whwh_bs[idx[0]]
        src_boxes_ = src_boxes / aug_whwh_bs
        target_boxes_ = target_boxes / aug_whwh_bs
        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction="none")
        losses["loss_bbox"] = (
            loss_bbox.sum() / num_boxes * self.weight_dict["loss_bbox"]
        )

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def det_losses(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        mt_indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )

        # For dist only
        comm.synchronize()
        all_num_boxes = comm.all_gather(num_boxes)
        num_boxes = sum([num.to("cpu") for num in all_num_boxes])
        num_boxes = torch.clamp(num_boxes / comm.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            ld = self.get_loss(loss, outputs, targets, mt_indices, num_boxes, **kwargs)
            losses.update(ld)
        match_inds = []
        aux_losses = {}
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            # sum all aux loss for better visualization
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                match_inds.append(indices)
                for loss in self.losses:
                    if loss == "cardinality":
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs["log"] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    aux_losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs
                )
                l_dict = {k + f"_enc": v for k, v in l_dict.items()}
                losses.update({k: v * self.weight_dict[k] for k, v in l_dict.items()})
        match_inds.append(mt_indices)
        return match_inds, losses, aux_losses

    def forward(self, outputs, targets):
        match_inds, losses, aux_losses = self.det_losses(outputs, targets)
        losses.update(aux_losses)
        return match_inds, losses


class DDetrSetCriterion(SrcnnSetCriterion):
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        # xyxy_abs -> ccwh_rel
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        aug_whwh_bs = torch.cat([v["aug_whwh"] for v in targets], dim=0)
        losses = {}
        loss_giou = 1 - torch.diag(
            iou_tools.generalized_box_iou(src_boxes, target_boxes)
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        aug_whwh_bs = aug_whwh_bs[idx[0]]
        src_boxes_ = src_boxes / aug_whwh_bs  # xyxy_rel
        target_boxes_ = target_boxes / aug_whwh_bs  # xyxy_rel
        src_boxes_ = iou_tools.box_xyxy_to_cxcywh(src_boxes_)  # ccwh_rel
        target_boxes_ = iou_tools.box_xyxy_to_cxcywh(target_boxes_)  # ccwh_rel
        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction="none")
        losses["loss_bbox"] = (
            loss_bbox.sum() / num_boxes * self.weight_dict["loss_bbox"]
        )

        return losses
    


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss
)  # type: torch.jit.ScriptModule

"""
class TridSrcnnSetCriterion(OIMSrcnnSetCriterion):
    def __init__(
        self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal
    ):
        super().__init__(
            cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal
        )
        self.det_ada = cfg.MODEL.SEARCH.PERSON_FEAT.SCORE_ADA != "null"

    def loss_oim(self, reid_feats, fl_feat_ids, fl_mms=None, fl_det_scores=None):
        
        Args:
            reid_feats: BN x C,
            feat_ids: BN,
        
        t2, wlun = 0.5, 0.1
        lb_matching_layer = self.labeled_matching_layer
        ulb_matching_layer = self.unlabeled_matching_layer
        # save for visualization
        pos_inds = (
            (fl_feat_ids > -3).nonzero().reshape(-1)
        )  # do not filter out background in case pos_inds become empty
        c = reid_feats.shape[-1]
        fl_reid_feats = reid_feats.view(-1, c)
        pos_reid = fl_reid_feats[pos_inds]
        pos_reid = F.normalize(pos_reid)
        pos_reid_ids = fl_feat_ids[pos_inds]
        if fl_mms is not None:
            mms = fl_mms[pos_inds] * MM_FACTOR  # TODO mm_factor for stability
        else:
            mms = None
        labeled_matching_scores: Tensor = lb_matching_layer(pos_reid, pos_reid_ids, mms)

        unlabeled_matching_scores = ulb_matching_layer(pos_reid, pos_reid_ids)

        if fl_det_scores is not None:
            labeled_matching_scores = labeled_matching_scores * fl_det_scores[:, None]
            unlabeled_matching_scores = (
                unlabeled_matching_scores * fl_det_scores[:, None]
            )

        lb_mask = fl_feat_ids > -1
        ulb_mask = fl_feat_ids == -1
        n_labeled = lb_mask.sum()
        n_unlabeled = ulb_mask.sum()
        b_ids = labeled_matching_scores.new_tensor(
            list(range(n_labeled)), dtype=torch.long
        )
        lb_gt_matching_scores = labeled_matching_scores[lb_mask][
            (b_ids, fl_feat_ids[lb_mask])
        ]
        sync_data = (n_labeled, n_unlabeled, lb_gt_matching_scores.detach())
        comm.synchronize()
        all_num_lbulb_lbgt = comm.all_gather(sync_data)
        sum_lb, sum_ulb = 0, 0
        lb_gt_mt_scores = []
        for nlb, nulb, mt_scores in all_num_lbulb_lbgt:
            sum_lb += nlb.cpu()
            sum_ulb += nulb.cpu()
            lb_gt_mt_scores.append(mt_scores.to(lb_gt_matching_scores.device))
        num_lb = torch.clamp(sum_lb / comm.get_world_size(), min=1).item()
        num_ulb = torch.clamp(sum_ulb / comm.get_world_size(), min=1).item()
        all_lb_gt_mt_scores = torch.cat(lb_gt_mt_scores, dim=0)
        # loss bin
        if lb_gt_matching_scores.shape[0] == 0:
            loss_bin = lb_gt_matching_scores.new_tensor([0])
        else:
            loss_bin = (
                1 + ((1 - lb_gt_matching_scores) * t2).exp()
            ).log()  # (num_lb, )
        # loss unlabeled TODO sync but no gradient to mean score item
        ulb_matching_scores = labeled_matching_scores[ulb_mask]
        if ulb_matching_scores.shape[0] == 0 or all_lb_gt_mt_scores.shape[0] == 0:
            loss_un = ulb_matching_scores.new_tensor([0])
        else:

            loss_un = (
                1
                + (
                    (ulb_matching_scores - all_lb_gt_mt_scores.mean())
                    * self.unlabel_weight
                )
                .exp()
                .sum(-1)
            ).log()  # (num_ulb, )
        # loss id
        id_lb_matching_scores = labeled_matching_scores * self.temperature
        id_ulb_matching_scores = unlabeled_matching_scores * self.unlabel_weight
        id_matching_scores = torch.cat(
            (id_lb_matching_scores, id_ulb_matching_scores), dim=1
        )  # + MINI
        pid_labels = pos_reid_ids.clone()
        pid_labels[pid_labels == -2] = -1
        loss_id = F.cross_entropy(
            id_matching_scores, pid_labels, reduction="none", ignore_index=-1
        )
        return {
            "loss_oim": loss_id.sum() / num_lb,
            "loss_bin": loss_bin.sum() / num_lb,
            "loss_un": loss_un.sum() / num_ulb * wlun,
        }

    def update_with_search_losses(self, outputs, targets, losses, **kw):
        
        gt_feats_ids:
            {"reid_feats": flattened feats,
            "ids": flattened ids}
        
        # TODO iou score instead of det score
        assign_base = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }
        gt_feats_ids = kw.pop("gt_feats_ids", None)
        if gt_feats_ids is not None:
            fl_gt_feats = gt_feats_ids["reid_feats"]
            fl_gt_ids = gt_feats_ids["ids"]

        if self.aux_loss_ps:
            inter_outs = outputs["aux_outputs"]
            if self.aux_shared:
                ids_lvl, mms_lvl, reid_feats_lvl, ada_scores_lvl = [], [], [], []
                for i in range(len(inter_outs)):
                    if self.det_ada:
                        ids, mms, bt_ious = self._id_assign_method(
                            assign_base, targets, None, is_ccwh=False, return_iou=True
                        )
                        if self.sim_score_ada == "iou":
                            fl_det_scores = bt_ious.view(-1)
                        else:
                            det_obj = assign_base["pred_logits"].sigmoid()
                            if self.sim_score_ada == "obj":
                                fl_det_scores = det_obj.detach().view(-1)
                            elif self.sim_score_ada == "obj_bp":
                                fl_det_scores = det_obj.view(-1)
                    else:
                        ids, mms = self._id_assign_method(
                            inter_outs[i], targets, None, is_ccwh=False
                        )
                    ids_lvl.append(ids.view(-1))
                    if mms is not None:
                        fl_mms = mms.view(-1)
                        mms_lvl.append(fl_mms)
                    reid_feats = inter_outs[i]["reid_feats"]
                    c = reid_feats.shape[-1]
                    reid_feats = reid_feats.reshape(-1, c)
                    reid_feats_lvl.append(reid_feats)
                    if self.det_ada:
                        ada_scores_lvl.append(fl_det_scores)
                if self.det_ada:
                    ids, mms, bt_ious = self._id_assign_method(
                        assign_base, targets, None, is_ccwh=False, return_iou=True
                    )
                    if self.sim_score_ada == "iou":
                        fl_det_scores = bt_ious.view(-1)
                    else:
                        det_obj = assign_base["pred_logits"].sigmoid()
                        if self.sim_score_ada == "obj":
                            fl_det_scores = det_obj.detach().view(-1)
                        elif self.sim_score_ada == "obj_bp":
                            fl_det_scores = det_obj.view(-1)
                else:
                    ids, mms = self._id_assign_method(
                        assign_base, targets, None, is_ccwh=False
                    )
                # save for visualization
                outputs["assign_ids"] = ids
                ids_lvl.append(ids.view(-1))
                if mms is not None:
                    fl_mms = mms.view(-1)
                    mms_lvl.append(fl_mms)
                reid_feats = assign_base["reid_feats"]
                c = reid_feats.shape[-1]
                reid_feats = reid_feats.reshape(-1, c)
                reid_feats_lvl.append(reid_feats)
                reid_feats = torch.cat(reid_feats_lvl, dim=0)
                fl_ids = torch.cat(ids_lvl, dim=0)
                if self.det_ada:
                    ada_scores_lvl.append(fl_det_scores)
                    fl_det_scores = torch.cat(ada_scores_lvl, dim=0)
                else:
                    fl_det_scores = None
                if len(mms_lvl) > 0:
                    fl_mms = torch.cat(mms_lvl, dim=0)
                else:
                    fl_mms = None
                if gt_feats_ids is not None:
                    reid_feats = torch.cat([reid_feats, fl_gt_feats], dim=0)
                    fl_ids = torch.cat([fl_ids, fl_gt_ids], dim=0)
                    if fl_mms is not None:
                        fl_mms = torch.cat(
                            [fl_mms, fl_mms.new_ones(fl_gt_ids.shape[0])], dim=0
                        )
                    if fl_det_scores is not None:
                        fl_det_scores = torch.cat(
                            [fl_det_scores, fl_det_scores.new_ones(fl_gt_ids.shape[0])],
                            dim=0,
                        )
                loss = self.loss_oim(reid_feats, fl_ids, fl_mms, fl_det_scores, **kw)
                losses.update(loss)
                return
            for i in range(len(inter_outs)):
                assign_base_inter = inter_outs[i]
                if self.det_ada:
                    ids, mms, bt_ious = self._id_assign_method(
                        assign_base_inter, targets, None, is_ccwh=False, return_iou=True
                    )
                    if self.sim_score_ada == "iou":
                        fl_det_scores = bt_ious.view(-1)
                    else:
                        det_obj = assign_base_inter["pred_logits"].sigmoid()
                        if self.sim_score_ada == "obj":
                            fl_det_scores = det_obj.detach().view(-1)
                        elif self.sim_score_ada == "obj_bp":
                            fl_det_scores = det_obj.view(-1)
                else:
                    ids, mms = self._id_assign_method(
                        assign_base_inter, targets, None, is_ccwh=False
                    )
                reid_feats = assign_base_inter["reid_feats"]
                c = reid_feats.shape[-1]
                reid_feats = reid_feats.reshape(-1, c)
                fl_ids = ids.view(-1)
                if mms is not None:
                    fl_mms = mms.view(-1)
                else:
                    fl_mms = None
                if gt_feats_ids is not None:
                    reid_feats = torch.cat([reid_feats, fl_gt_feats], dim=0)
                    fl_ids = torch.cat([fl_ids, fl_gt_ids], dim=0)
                    if fl_mms is not None:
                        fl_mms = torch.cat(
                            [fl_mms, fl_mms.new_ones(fl_gt_ids.shape[0])], dim=0
                        )
                    if self.det_ada:
                        fl_det_scores = torch.cat(
                            [fl_det_scores, fl_det_scores.new_ones(fl_gt_ids.shape[0])],
                            dim=0,
                        )
                if not self.det_ada:
                    fl_det_scores = None
                loss = self.loss_oim(reid_feats, fl_ids, fl_mms, fl_det_scores, **kw)
                rsts = {}
                for k in loss:
                    rsts[k + "_{}".format(i)] = loss[k]
                losses.update(rsts)

        if self.det_ada:
            ids, mms, bt_ious = self._id_assign_method(
                assign_base, targets, None, is_ccwh=False, return_iou=True
            )
            if self.sim_score_ada == "iou":
                fl_det_scores = bt_ious.view(-1)
            else:
                det_obj = assign_base["pred_logits"].sigmoid()
                if self.sim_score_ada == "obj":
                    fl_det_scores = det_obj.detach().view(-1)
                elif self.sim_score_ada == "obj_bp":
                    fl_det_scores = det_obj.view(-1)

        else:
            ids, mms = self._id_assign_method(assign_base, targets, None, is_ccwh=False)
        # save for visualization
        outputs["assign_ids"] = ids
        reid_feats = assign_base["reid_feats"]
        c = reid_feats.shape[-1]
        reid_feats = reid_feats.reshape(-1, c)
        fl_ids = ids.view(-1)
        if mms is not None:
            fl_mms = mms.view(-1)
        else:
            fl_mms = None
        if gt_feats_ids is not None:
            reid_feats = torch.cat([reid_feats, fl_gt_feats], dim=0)
            fl_ids = torch.cat([fl_ids, fl_gt_ids], dim=0)
            if fl_mms is not None:
                fl_mms = torch.cat([fl_mms, fl_mms.new_ones(fl_gt_ids.shape[0])], dim=0)
            if self.det_ada:
                fl_det_scores = torch.cat(
                    [fl_det_scores, fl_det_scores.new_ones(fl_gt_ids.shape[0])],
                    dim=0,
                )
        if not self.det_ada:
            fl_det_scores = None
        loss = self.loss_oim(reid_feats, fl_ids, fl_mms, fl_det_scores, **kw)
        losses.update(loss)

"""
