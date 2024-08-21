# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import torch

from psd2.layers import nonzero_tuple


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.

    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.

    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(
        self,
        thresholds: List[float],
        labels: List[int],
        allow_low_quality_matches: bool = False,
    ):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.

            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        # Add -inf and +inf to first and last position in thresholds
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        # Currently torchscript does not support all + generator
        assert all(
            [low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])]
        )
        assert all([l in [-1, 0, 1] for l in labels])
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )
            # When no gt boxes exist, we define IOU = 0 and therefore set labels
            # to `self.labels[0]`, which usually defaults to background class 0
            # To choose to ignore instead, can make labels=[-1,0,-1,1] + set appropriate thresholds
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(
            self.labels, self.thresholds[:-1], self.thresholds[1:]
        ):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of
        :paper:`Faster R-CNN`.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        _, pred_inds_with_highest_quality = nonzero_tuple(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # If an anchor was labeled positive only due to a low-quality match
        # with gt_A, but it has larger overlap with gt_B, it's matched index will still be gt_B.
        # This follows the implementation in Detectron, and is found to have no significant impact.
        match_labels[pred_inds_with_highest_quality] = 1


# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from psd2.structures.boxes import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    box_xyxy_to_cxcywh,
)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class=None, cost_bbox=None, cost_giou=None):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        if cost_class is None:
            return
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"
        self._match_costs = {}

    def get_cost(self, cost_name):
        return (
            self._match_costs[cost_name]["cost"],
            self._match_costs[cost_name]["weight"],
        )

    def _cls_cost(self, outputs, targets, bs, nq, sizes):
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        cost_class_bs = cost_class.view(bs, nq, -1)
        cost_class_bs = [c[i] for i, c in enumerate(cost_class_bs.split(sizes, -1))]
        self._match_costs["class"] = {
            "cost": cost_class_bs,
            "weight": self.cost_class,
        }
        return cost_class_bs

    def _bbox_cost(self, outputs, targets, bs, nq, sizes):
        # (0,1) ccwh in aug image
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_bbox_bs = cost_bbox.view(bs, nq, -1)
        cost_bbox_bs = [c[i] for i, c in enumerate(cost_bbox_bs.split(sizes, -1))]
        self._match_costs["bbox"] = {"cost": cost_bbox_bs, "weight": self.cost_bbox}
        return cost_bbox_bs

    def _giou_cost(self, outputs, targets, bs, nq, sizes, is_ccwh=True):
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # Compute the giou cost betwen boxes
        if is_ccwh:
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
            )
        else:
            cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        cost_giou_bs = cost_giou.view(bs, nq, -1)
        cost_giou_bs = [c[i] for i, c in enumerate(cost_giou_bs.split(sizes, -1))]
        self._match_costs["giou"] = {"cost": cost_giou_bs, "weight": self.cost_giou}
        return cost_giou_bs

    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            sizes = [len(v["boxes"]) for v in targets]

            cost_class_bs = self._cls_cost(outputs, targets, bs, num_queries, sizes)
            cost_bbox_bs = self._bbox_cost(outputs, targets, bs, num_queries, sizes)
            cost_giou_bs = self._giou_cost(outputs, targets, bs, num_queries, sizes)

            # Final cost matrix
            C = [
                self.cost_bbox * cost_bbox_bs[i]
                + self.cost_class * cost_class_bs[i]
                + self.cost_giou * cost_giou_bs[i]
                for i in range(bs)
            ]
            # C = C.view(bs, num_queries, -1).cpu()

            indices = [linear_sum_assignment(c.cpu()) for c in C]
            return [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices
            ]


EPS = 1e-10


class OlnHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_score: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        """Creates the matcher

        Params:
            cost_score: This is the relative weight of the OLN score error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_score = cost_score
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_score != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cannot be 0"
        self._match_costs = {}

    def get_cost(self, cost_name):
        return (
            self._match_costs[cost_name]["cost"],
            self._match_costs[cost_name]["weight"],
        )

    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(
                0, 1
            )  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            # tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            sizes = [len(v["boxes"]) for v in targets]
            # Compute the centerness cost.
            """alpha = 0.25
            gamma = 2.0
            neg_cost_class = (
                (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]"""
            rpts = outputs["pred_ref_pts"]
            rx = rpts[..., 0].view(-1, 1)  # bn x 1
            ry = rpts[..., 1].view(-1, 1)  # bn x 1
            xc = tgt_bbox[:, 0].unsqueeze(0)  # 1 x g
            yc = tgt_bbox[:, 1].unsqueeze(0)  # 1 x g
            gw = tgt_bbox[:, 2].unsqueeze(0)  # 1 x g
            gh = tgt_bbox[:, 3].unsqueeze(0)  # 1 x g
            l_ct = rx - (xc - gw / 2)  # bn x g
            r_ct = (xc + gw / 2) - rx  # bn x g
            t_ct = (yc + gh / 2) - ry  # bn x g
            b_ct = ry - (yc - gh / 2)  # bn x g
            ct_sq = (
                torch.where(l_ct < r_ct, l_ct, r_ct)
                / torch.where(l_ct > r_ct, l_ct, r_ct)
                * torch.where(t_ct < b_ct, t_ct, b_ct)
                / torch.where(t_ct > b_ct, t_ct, b_ct)
            )  # bn x g
            out_box_mask = l_ct <= 0 or r_ct <= 0 or t_ct <= 0 or b_ct <= 0
            ct_sq[out_box_mask] = EPS
            pairwise_ct = ct_sq ** (0.5)
            cost_score = -pairwise_ct.log()
            cost_score_bs = cost_score.view(bs, num_queries, -1)
            cost_score_bs = [c[i] for i, c in enumerate(cost_score_bs.split(sizes, -1))]
            self._match_costs["score"] = {
                "cost": cost_score_bs,
                "weight": self.cost_score,
            }
            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_bbox_bs = cost_bbox.view(bs, num_queries, -1)
            cost_bbox_bs = [c[i] for i, c in enumerate(cost_bbox_bs.split(sizes, -1))]
            self._match_costs["bbox"] = {"cost": cost_bbox_bs, "weight": self.cost_bbox}
            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
            )
            cost_giou_bs = cost_giou.view(bs, num_queries, -1)
            cost_giou_bs = [c[i] for i, c in enumerate(cost_giou_bs.split(sizes, -1))]
            self._match_costs["giou"] = {"cost": cost_giou_bs, "weight": self.cost_giou}
            # Final cost matrix
            """C = (
                self.cost_bbox * cost_bbox
                + self.cost_score * cost_score
                + self.cost_giou * cost_giou
            )
            C = C.view(bs, num_queries, -1).cpu()"""
            C = [
                self.cost_bbox * cost_bbox_bs[i]
                + self.cost_score * cost_score_bs[i]
                + self.cost_giou * cost_giou_bs[i]
                for i in range(bs)
            ]

            indices = [linear_sum_assignment(c.cpu()) for c in C]
            return [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices
            ]


class SrcnnHungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        use_focal: bool = False,
        focal_alpha=0.25,
        focal_gamma=2,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        if self.use_focal:
            self.focal_loss_alpha = focal_alpha
            self.focal_loss_gamma = focal_gamma
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"
        self._match_costs = {}

    def _cls_cost(self, outputs, targets, bs, nq, sizes):
        # We flatten to compute the cost matrices in a batch
        if self.use_focal:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).sigmoid()
            )  # [batch_size * num_queries, num_classes]
        else:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal:
            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (
                (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]
        cost_class_bs = cost_class.view(bs, nq, -1)
        cost_class_bs = [c[i] for i, c in enumerate(cost_class_bs.split(sizes, -1))]
        self._match_costs["class"] = {
            "cost": cost_class_bs,
            "weight": self.cost_class,
        }
        return cost_class_bs

    def _bbox_cost(self, outputs, targets, bs, nq, sizes):
        aug_whwh_bs = torch.stack([v["aug_whwh"] for v in targets])
        out_bbox = outputs["pred_boxes"] / aug_whwh_bs  # [batch_size , num_queries, 4]

        out_bbox = out_bbox.flatten(0, 1)  # (0,1) xyxy in aug image
        tgt_bbox = torch.cat([v["boxes"] / v["aug_whwh"] for v in targets])
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_bbox_bs = cost_bbox.view(bs, nq, -1)
        cost_bbox_bs = [c[i] for i, c in enumerate(cost_bbox_bs.split(sizes, -1))]
        self._match_costs["bbox"] = {"cost": cost_bbox_bs, "weight": self.cost_bbox}
        return cost_bbox_bs

    def _giou_cost(self, outputs, targets, bs, nq, sizes):
        out_bbox = outputs["pred_boxes"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]# x,y,x,y  of aug image
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # x,y,x,y  of aug image
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        cost_giou_bs = cost_giou.view(bs, nq, -1)
        cost_giou_bs = [c[i] for i, c in enumerate(cost_giou_bs.split(sizes, -1))]
        self._match_costs["giou"] = {"cost": cost_giou_bs, "weight": self.cost_giou}
        return cost_giou_bs

    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            sizes = [len(v["boxes"]) for v in targets]

            cost_class_bs = self._cls_cost(outputs, targets, bs, num_queries, sizes)
            cost_bbox_bs = self._bbox_cost(outputs, targets, bs, num_queries, sizes)
            cost_giou_bs = self._giou_cost(outputs, targets, bs, num_queries, sizes)

            # Final cost matrix
            C = [
                self.cost_bbox * cost_bbox_bs[i]
                + self.cost_class * cost_class_bs[i]
                + self.cost_giou * cost_giou_bs[i]
                for i in range(bs)
            ]
            # C = C.view(bs, num_queries, -1).cpu()

            indices = [linear_sum_assignment(c.cpu()) for c in C]
            return [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices
            ]


class DDetrHungarianMatcher(SrcnnHungarianMatcher):
    def _bbox_cost(self, outputs, targets, bs, nq, sizes):
        # xyxy_abs -> ccwh_rel

        # (0,1) ccwh in aug image
        aug_whwh_bs = torch.stack([v["aug_whwh"] for v in targets])
        out_bbox = outputs["pred_boxes"] / aug_whwh_bs  # xyxy_rel
        out_bbox = out_bbox.flatten(0, 1)  # [batch_size * num_queries, 4]
        out_box = box_xyxy_to_cxcywh(out_bbox)  # ccwh_rel

        tgt_bbox = torch.cat([v["boxes"] / v["aug_whwh"] for v in targets])  # xyxy_rel
        tgt_bbox = box_xyxy_to_cxcywh(tgt_bbox)  # ccwh_rel
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_bbox_bs = cost_bbox.view(bs, nq, -1)
        cost_bbox_bs = [c[i] for i, c in enumerate(cost_bbox_bs.split(sizes, -1))]
        self._match_costs["bbox"] = {"cost": cost_bbox_bs, "weight": self.cost_bbox}
        return cost_bbox_bs


class MinCostMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        head_type,
        focal_alpha,
        focal_gamma,
        pre_define,
        sizes_of_interest,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss_alpha = focal_alpha
        self.focal_loss_gamma = focal_gamma
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"
        self.head_type = head_type
        assert self.head_type in ["center", "fcos", "retina"]
        self.pre_define = pre_define
        self.object_sizes_of_interest = torch.tensor(sizes_of_interest)

    @torch.no_grad()
    def compute_cost_fpn(self, cost_bbox, tgt_bbox, out_bbox, fpn_level):
        """
        cost_bbox: (nr_out, nr_tgt)
        tgt_bbox:  (nr_tgt, 4)
        out_bbox:  (nr_out, 4), 4 is (x y x y)
        fpn_level: (nr_out,)
        object_sizes_of_interest: (nr_out, 2)
        """
        object_sizes_of_interest = self.object_sizes_of_interest[fpn_level.long()].to(
            cost_bbox.device
        )

        left = out_bbox[:, 0, None] - tgt_bbox[None, :, 0]
        top = out_bbox[:, 1, None] - tgt_bbox[None, :, 1]
        right = tgt_bbox[None, :, 2] - out_bbox[:, 2, None]
        top = tgt_bbox[None, :, 3] - out_bbox[:, 3, None]
        distances = torch.stack([left, top, right, top], dim=-1)

        max_distances = distances.max(dim=-1)[0]
        is_cared_in_the_level = (
            max_distances >= object_sizes_of_interest[:, None, 0]
        ) & (max_distances <= object_sizes_of_interest[:, None, 1])

        cost_fpn = (1.0 - is_cared_in_the_level.float()) * 9999
        return cost_fpn

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # NOTE output flatten is performed in model forward
        if self.head_type == "center":
            """
            bs, k, h, w = outputs["pred_logits"].shape
            hw = h * w
            # We flatten to compute the cost matrices in a batch
            batch_out_prob = (
                outputs["pred_logits"]
                .permute(0, 2, 3, 1)
                .reshape(bs, h * w, k)
                .sigmoid()
            )  # [batch_size, num_queries, num_classes]
            """
            batch_out_prob = outputs["pred_logits"].sigmoid()
            if self.pre_define:
                """
                pre_locations = outputs["locations"] # bs 2 h w
                pre_boxes = torch.cat([pre_locations, pre_locations], dim=1)#
                batch_out_bbox = pre_boxes.permute(0, 2, 3, 1).reshape(
                    bs, h * w, 4
                )  # [batch_size, num_queries, 4]
                """
                pre_locations = outputs["locations"]  # bs hw 2
                batch_out_bbox = torch.cat([pre_locations, pre_locations], dim=-1)

            else:
                """
                batch_out_bbox = (
                    outputs["pred_boxes"].permute(0, 2, 3, 1).reshape(bs, h * w, 4)
                )  # [batch_size, num_queries, 4]
                """
                batch_out_bbox = outputs["pred_boxes"]

        elif self.head_type == "retina":
            """
            bs, k, hw = outputs["pred_logits"].shape
            # We flatten to compute the cost matrices in a batch
            batch_out_prob = (
                outputs["pred_logits"].permute(0, 2, 1).sigmoid()
            )  # [batch_size, num_queries, num_classes]
            """
            batch_out_prob = outputs["pred_logits"].sigmoid()
            if self.pre_define:
                """batch_out_bbox = outputs["anchors"].permute(
                    0, 2, 1
                )  # [batch_size, num_queries, 4]"""
                batch_out_bbox = outputs["anchors"]
            else:
                """batch_out_bbox = outputs["pred_boxes"].permute(
                    0, 2, 1
                )  # [batch_size, num_queries, 4]"""
                batch_out_bbox = outputs["pred_boxes"]

        else:  # 'FCOS'
            """bs, k, hw = outputs["pred_logits"].shape
            # We flatten to compute the cost matrices in a batch
            batch_out_prob = (
                outputs["pred_logits"].permute(0, 2, 1).sigmoid()
            )  # [batch_size, num_queries, num_classes]"""
            batch_out_prob = outputs["pred_logits"].sigmoid()
            if self.pre_define:
                """pre_locations = outputs["locations"]
                pre_boxes = torch.cat([pre_locations, pre_locations], dim=1)
                batch_out_bbox = pre_boxes.permute(
                    0, 2, 1
                )  # [batch_size, num_queries, 4]"""
                pre_locations = outputs["locations"]  # bs hw 2
                batch_out_bbox = torch.cat([pre_locations, pre_locations], dim=-1)
                batch_fpn_levels = outputs["fpn_levels"]
            else:
                """batch_out_bbox = outputs["pred_boxes"].permute(
                    0, 2, 1
                )  # [batch_size, num_queries, 4]"""
                batch_out_bbox = outputs["pred_boxes"]
        indices = []
        bs, hw = batch_out_prob.shape[:2]
        for i in range(bs):
            tgt_ids = targets[i]["labels"]

            if tgt_ids.shape[0] == 0:
                #                 indices.append((torch.as_tensor([]).to(batch_out_prob), torch.as_tensor([]).to(batch_out_prob)))
                indices.append((torch.as_tensor([]), torch.as_tensor([])))
                continue

            tgt_bbox = targets[i]["boxes"]
            out_prob = batch_out_prob[i]
            out_bbox = batch_out_bbox[i]

            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (
                (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            """image_size_out = targets[i]["image_size_xyxy"].unsqueeze(0).repeat(hw, 1)
            image_size_tgt = targets[i]["image_size_xyxy_tgt"]

            out_bbox_ = out_bbox / image_size_out
            tgt_bbox_ = tgt_bbox / image_size_tgt"""
            aug_whwh_bs = targets[i]["aug_whwh"]  # 1 x 4
            out_bbox_ = out_bbox / aug_whwh_bs  # [ num_queries, 4]
            tgt_bbox_ = targets[i]["boxes"] / aug_whwh_bs  # [num_gts, 4]

            if self.pre_define:
                if self.head_type == "center":
                    # comparison center distance
                    cost_bbox = torch.cdist(
                        out_bbox_[:, :2] + out_bbox_[:, 2:],
                        tgt_bbox_[:, :2] + tgt_bbox_[:, 2:],
                        p=1,
                    )
                    C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
                elif self.head_type == "retina":
                    # Compute the L1 cost between boxes
                    cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)
                    # Compute the giou cost betwen boxes
                    cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
                    C = (
                        self.cost_bbox * cost_bbox
                        + self.cost_class * cost_class
                        + self.cost_giou * cost_giou
                    )
                else:  #  'FCOS'
                    # comparison center distance
                    cost_bbox = torch.cdist(
                        out_bbox_[:, :2] + out_bbox_[:, 2:],
                        tgt_bbox_[:, :2] + tgt_bbox_[:, 2:],
                        p=1,
                    )
                    cost_fpn = self.compute_cost_fpn(
                        cost_bbox, tgt_bbox, out_bbox, batch_fpn_levels[i][0]
                    )
                    C = (
                        self.cost_bbox * cost_bbox
                        + self.cost_class * cost_class
                        + cost_fpn
                    )
            else:
                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)
                # Compute the giou cost betwen boxes
                cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
                C = (
                    self.cost_bbox * cost_bbox
                    + self.cost_class * cost_class
                    + self.cost_giou * cost_giou
                )

            #             _, src_ind = torch.min(C, dim=0)
            #             tgt_ind = torch.arange(len(tgt_ids)).to(src_ind)
            src_ind, tgt_ind = linear_sum_assignment(C.cpu())
            indices.append((src_ind, tgt_ind))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
