# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import List
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from psd2.config import configurable
from psd2.structures import ImageList, Instances
from psd2.utils.events import get_event_storage


from ..backbone import build_backbone
from ..box_regression import _dense_box_regression_loss
from .build import META_ARCH_REGISTRY
from .retinanet import RetinaNet, RetinaNetHead


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class RetinaNetM(RetinaNet):
    """
    Implement RetinaNet with softmax.
    """

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        head = RetinaNetHeadM(cfg, feature_shapes)
        ret["head"] = head
        return ret

    def forward_training(self, images, features, predictions, gt_instances):
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
            predictions, [self.num_classes + 1, 4]
        )
        anchors = self.anchor_generator(features)
        gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
        return self.losses(
            anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes
        )

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor storing the loss.
                Used during training only. The dict keys are: "loss_cls" and "loss_box_reg"
        """
        # TODO check if split pos mask for cls and reg
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)

        valid_mask = gt_labels >= 0
        """
        pos_mask_cls = (
            gt_labels >= 0
        )  # & (gt_labels != self.num_classes), NOTE bg is valid class now
        pos_mask_box = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors_cls = pos_mask_cls.sum().item()
        num_pos_anchors_box = pos_mask_box.sum().item()
        normalizer_cls = self._ema_update(
            "loss_normalizer_cls", max(num_pos_anchors_cls, 1), 100
        )
        normalizer_box = self._ema_update(
            "loss_normalizer_box", max(num_pos_anchors_box, 1), 100
        )
        """
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 100)

        # classification and regression loss

        gt_labels_target = F.one_hot(
            gt_labels[valid_mask], num_classes=self.num_classes + 1
        )[..., :-1]

        # no sigmoid
        p = F.softmax(torch.cat(pred_logits, dim=1), dim=-1)[valid_mask][..., :-1]
        targets = gt_labels_target.to(pred_logits[0].dtype)
        ce_loss = F.binary_cross_entropy(
            p,
            targets,
            reduction="none",
        )
        alpha = self.focal_loss_alpha
        gamma = self.focal_loss_gamma

        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        loss_cls = loss.sum()

        loss_box_reg = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        return {
            "loss_cls": loss_cls / normalizer,
            "loss_box_reg": loss_box_reg / normalizer,
        }

    def forward_inference(
        self, images: ImageList, features: List[Tensor], predictions: List[List[Tensor]]
    ):
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
            predictions, [self.num_classes + 1, 4]
        )
        anchors = self.anchor_generator(features)

        results: List[Instances] = []
        for img_idx, image_size in enumerate(images.image_sizes):
            scores_per_image = [
                F.softmax(x[img_idx], dim=-1)[..., :-1] for x in pred_logits
            ]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors, scores_per_image, deltas_per_image, image_size
            )
            results.append(results_per_image)
        return results


class RetinaNetHeadM(RetinaNetHead):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    @configurable
    def __init__(self, *args, **kws):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors
            conv_dims (List[int]): dimensions for each convolution layer
            norm (str or callable):
                Normalization for conv layers except for the two output layers.
                See :func:`detectron2.layers.get_norm` for supported types.
            prior_prob (float): Prior weight for computing bias
        """
        super().__init__(*args, **kws)
        conv_dims: List[int] = kws["conv_dims"]
        num_anchors = kws["num_anchors"]
        num_classes = kws["num_classes"]
        prior_prob = kws["prior_prob"]
        self.cls_score = nn.Conv2d(
            conv_dims[-1],
            num_anchors * (num_classes + 1),  # NOTE add background class
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Initialization
        for layer in self.cls_score.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)
