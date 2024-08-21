from typing import Iterable
import torch
from abc import ABCMeta, abstractmethod
import torchvision


class IdAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(self, det_bboxes, det_scores, gt_bboxes, gt_labels):
        """Assign boxes to either a ground truth boxes or a negative boxes."""


class IouIDAssigner(IdAssigner):
    def __init__(self, iou_t) -> None:
        self.iou_t = iou_t

    def assign(self, det_bboxes, det_scores, gt_bboxes, gt_labels):
        """
        boxes: xyxy
        """
        if isinstance(det_bboxes, torch.Tensor):
            bs, ns = det_bboxes.shape[:2]
            output_ids = (
                torch.zeros(bs, ns, dtype=torch.long, device=det_bboxes[0].device) - 2
            )
        else:
            assert isinstance(det_bboxes, Iterable)
            bs = len(det_bboxes)
            output_ids = [
                torch.zeros(bi.shape[0], dtype=torch.long, device=det_bboxes[0].device)
                - 2
                for bi in det_bboxes
            ]
        for bi, gt_bboxes_i in enumerate(gt_bboxes):
            pred_bboxes = det_bboxes[bi]  # np x 4, x-y-x-y
            pairwise_ious = torchvision.ops.box_iou(pred_bboxes, gt_bboxes_i)  # np x ng
            pairwise_ious[pairwise_ious < self.iou_t] = 0
            max_iou, max_iou_inds = pairwise_ious.max(dim=1)
            output_ids[bi] = gt_labels[bi][max_iou_inds]
            output_ids[bi][max_iou == 0] = -2
        return output_ids


class SiouIDAssigner(IdAssigner):
    def __init__(self, iou_t, score_t) -> None:
        self.iou_t = iou_t
        self.score_t = score_t

    def assign(self, det_bboxes, det_scores, gt_bboxes, gt_labels):
        """
        boxes: xyxy
        """
        if isinstance(det_bboxes, torch.Tensor):
            bs, ns = det_bboxes.shape[:2]
            output_ids = (
                torch.zeros(bs, ns, dtype=torch.long, device=det_bboxes[0].device) - 2
            )
        else:
            assert isinstance(det_bboxes, Iterable)
            bs = len(det_bboxes)
            output_ids = [
                torch.zeros(bi.shape[0], dtype=torch.long, device=det_bboxes[0].device)
                - 2
                for bi in det_bboxes
            ]
        for bi, gt_bboxes_i in enumerate(gt_bboxes):
            pred_bboxes = det_bboxes[bi]  # np x 4, x-y-x-y
            pairwise_ious = torchvision.ops.box_iou(pred_bboxes, gt_bboxes_i)  # np x ng
            pairwise_ious[pairwise_ious < self.iou_t] = 0
            max_iou, max_iou_inds = pairwise_ious.max(dim=1)
            output_ids[bi] = gt_labels[bi][max_iou_inds]
            output_ids[bi][max_iou == 0] = -2
            output_ids[bi][det_scores[bi] < self.score_t] = -2
        return output_ids


def build_id_assigner(ias_cfg):
    ias_type = ias_cfg.NAME
    if ias_type == "iou":
        return IouIDAssigner(ias_cfg.POS_IOU)
    elif ias_type == "siou":
        return SiouIDAssigner(ias_cfg.POS_IOU, ias_cfg.POS_SCORE)
    else:
        return None
