from collections import OrderedDict

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RPNHead,
    RegionProposalNetwork,
)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from psd2.org_model_lib.hoim.model.resnet_backbone import resnet_backbone
from psd2.org_model_lib.hoim.loss.hoim import HOIMLoss
from ..build import META_ARCH_REGISTRY
import logging
from psd2.structures.nested_tensor import nested_collate_fn
import os

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class HOIM(GeneralizedRCNN):
    def __init__(self, cfg):
        search_cfg = cfg.MODEL.SEARCH
        model_cfg = search_cfg.HOIM
        oim_cfg = search_cfg.OIM
        rpn_cfg = model_cfg.RPN
        box_cfg = model_cfg.BOX
        backbone, conv_head = resnet_backbone("resnet50", True)
        hoim_loss = HOIMLoss(
            256,
            oim_cfg.LUT_SIZE,
            oim_cfg.CQ_SIZE,
            oim_cfg.BG_Q_SIZE,
            oim_cfg.OIM_MOMENTUM,
            oim_cfg.OIM_SCALAR,
            omega_decay=0.99,
            dynamic_lambda=True,
        )
        coord_fc = CoordRegressor(2048, num_classes=2)
        num_classes = None
        box_predictor = coord_fc
        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError(
                    "num_classes should be None when box_predictor is specified"
                )
        else:
            if box_predictor is None:
                raise ValueError(
                    "num_classes should not be None when box_predictor"
                    "is not specified"
                )

        out_channels = backbone.out_channels
        rpn_anchor_generator = AnchorGenerator(
            (rpn_cfg.ANCHOR_SCALES,), (rpn_cfg.ANCHOR_RATIOS,)
        )
        rpn_head = None
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(
            training=rpn_cfg.RPN_PRE_NMS_TOP_N, testing=rpn_cfg.RPN_PRE_NMS_TOP_N_TEST
        )
        rpn_post_nms_top_n = dict(
            training=rpn_cfg.RPN_POST_NMS_TOP_N, testing=rpn_cfg.RPN_POST_NMS_TOP_N_TEST
        )

        rpn = self._set_rpn(
            rpn_anchor_generator,
            rpn_head,
            rpn_cfg.RPN_POSITIVE_OVERLAP,
            rpn_cfg.RPN_NEGATIVE_OVERLAP,
            rpn_cfg.RPN_BATCH_SIZE,
            rpn_cfg.RPN_FG_FRACTION,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_cfg.RPN_NMS_THRESH,
        )
        box_roi_pool = None
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
            )
        feat_head = conv_head
        if feat_head is None:
            raise ValueError("feat_head should be specified manually.")
            # resolution = box_roi_pool.output_size[0]
            # representation_size = 2048
            # # ConvHead should be part of the backbone
            # # feat_head = TwoMLPHead(
            # #     out_channels * resolution ** 2,
            # #     representation_size)
        box_predictor = coord_fc
        if box_predictor is None:
            box_predictor = CoordRegressor(2048, num_classes)
        embedding_head = None
        if embedding_head is None:
            embedding_head = ReIDEmbeddingProj(
                featmap_names=["feat_res4", "feat_res5"],
                in_channels=[1024, 2048],
                dim=256,
            )
        reid_loss = hoim_loss
        if reid_loss is None:
            reid_loss = HOIMLoss(256, oim_cfg.LUT_SIZE, oim_cfg.CQ_SIZE, 0.5, 30.0)

        roi_heads = self._set_roi_heads(
            embedding_head,
            reid_loss,
            box_roi_pool,
            feat_head,
            box_predictor,
            box_cfg.BG_THRESH_HI,
            box_cfg.BG_THRESH_LO,
            box_cfg.RCNN_BATCH_SIZE,
            box_cfg.FG_FRACTION,
            box_cfg.BOX_REGRESSION_WEIGHTS,
            box_cfg.FG_THRESH,
            box_cfg.NMS_TEST,
            rpn_cfg.RPN_POST_NMS_TOP_N,
        )
        transform = GeneralizedRCNNTransform(
            cfg.INPUT.MIN_SIZE_TRAIN[0],
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.MODEL.PIXEL_MEAN,
            cfg.MODEL.PIXEL_STD,
        )
        super(HOIM, self).__init__(backbone, rpn, roi_heads, transform)
        wgt_cfg = search_cfg.LOSS_WEIGHTS
        self.lw_rpn_reg = wgt_cfg.LW_RPN_REG
        # Loss weight of RPN classification
        self.lw_rpn_cls = wgt_cfg.LW_RPN_CLS
        # Loss weight of box regression
        self.lw_rcnn_reg = wgt_cfg.LW_RCNN_REG
        # Loss weight of box classification
        self.lw_rcnn_cls = wgt_cfg.LW_RCNN_CLS
        # Loss weight of box OIM (i.e. Online Instance Matching)
        self.lw_reid = wgt_cfg.LW_BOX_REID

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def _set_roi_heads(self, *args):
        return HOIMRoIHeads(*args)

    def _set_rpn(self, *args):
        return RegionProposalNetwork(*args)

    def ex_feat(self, images, targets, mode="det"):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result: (tuple(Tensor)): list of 1 x d embedding for the RoI of each image

        """
        if mode == "det":
            return self.ex_feat_by_roi_pooling(images, targets)
        elif mode == "reid":
            return self.ex_feat_by_img_crop(images, targets)

    def ex_feat_by_roi_pooling(self, images, targets):
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals = [x["boxes"] for x in targets]

        roi_pooled_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes
        )
        rcnn_features = self.roi_heads.feat_head(roi_pooled_features)
        embeddings = self.roi_heads.embedding_head(rcnn_features)
        return embeddings.split(1, 0)

    def ex_feat_by_img_crop(self, images, targets):
        assert len(images) == 1, "Only support batch_size 1 in this mode"

        images, targets = self.transform(images, targets)
        x1, y1, x2, y2 = map(lambda x: int(round(x)), targets[0]["boxes"][0].tolist())
        input_tensor = images.tensors[:, :, y1 : y2 + 1, x1 : x2 + 1]
        features = self.backbone(input_tensor)
        features = features.values()[0]
        rcnn_features = self.roi_heads.feat_head(features)
        embeddings = self.roi_heads.embedding_head(rcnn_features)
        return embeddings.split(1, 0)

    def preprocess_input(self, input_list):
        img_paths = []
        img_names = []
        img_ts = []
        img_boxes = []
        img_pids = []
        img_aug_hws = []
        img_org_hws = []
        img_org_boxes = []
        for input_dict in input_list:
            img_paths.append(input_dict["file_name"])
            img_names.append(input_dict["image_id"])
            img_ts.append(input_dict["image"].to(self.device))
            xyxy_box_t = torch.tensor(
                input_dict["boxes"],
                dtype=torch.float32,
                device=self.device,
            )  # xyxy
            img_boxes.append(xyxy_box_t)
            img_pids.append(input_dict["ids"])
            img_aug_hws.append((input_dict["height"], input_dict["width"]))
            img_org_hws.append((input_dict["org_height"], input_dict["org_width"]))
            img_org_boxes.append(torch.tensor(input_dict["org_boxes"]))
        batched_input = [
            img_ts,
            img_paths,
            img_names,
            img_boxes,
            img_pids,
            img_aug_hws,
            img_org_hws,
            img_org_boxes,
        ]
        return nested_collate_fn(batched_input)

    def inference(self, input_list):
        if "query" in input_list[0]:
            input_batches = self.preprocess_input([qd["query"] for qd in input_list])
            images = input_batches[0]
            org_hws = input_batches[6]
            targets = []
            for fname, imgid, bboxes, ids in zip(*input_batches[1:5]):
                pids = images.tensors.new_tensor(ids, dtype=torch.int64)
                pids[pids > -1] += 1
                pids[pids == -1] = 5555
                targets.append(
                    {
                        "file_name": fname,
                        "image_id": imgid,
                        "boxes": bboxes,
                        "pid": pids,
                    }
                )
            features = self.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([(0, features)])
            proposals = [x["boxes"] for x in targets]

            roi_pooled_features = self.roi_heads.box_roi_pool(
                features, proposals, images.image_sizes
            )
            rcnn_features = self.roi_heads.feat_head(roi_pooled_features)
            embeddings = self.roi_heads.embedding_head(rcnn_features)
            p_feats = embeddings.cpu().split(1, 0)
            result_list = input_list.copy()
            for bi, feat in enumerate(p_feats):
                result_list[bi]["query"]["feat"] = feat.squeeze(0)
            return result_list
        else:
            input_batches = self.preprocess_input(input_list)
            images = input_batches[0]
            targets = None
            org_hws = input_batches[6]
            features = self.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])
            proposals, _ = self.rpn(images, features, targets)
            detections, _ = self.roi_heads(
                features, proposals, images.image_sizes, targets
            )
            detections = self.transform.postprocess(
                detections, images.image_sizes, org_hws
            )
            return_result = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for det in detections:
                return_result["pred_boxes"].append(det["boxes"].view(-1, 4))
                return_result["pred_scores"].append(det["scores"].view(-1, 1))
                return_result["reid_feats"].append(det["embeddings"].view(-1, 256))
            # vis_inf(input_batches, return_result, "outputs/test_debug", 0.0)
            return return_result

    def forward(self, input_list):
        if not self.training:
            return self.inference(input_list)
        input_batches = self.preprocess_input(input_list)
        images = input_batches[0].to(self.device)
        targets = []
        aug_hws = images.tensors.new_tensor(input_batches[5])
        org_hws = images.tensors.new_tensor(input_batches[6])
        for fname, imgid, bboxes, ids in zip(*input_batches[1:5]):
            pids = images.tensors.new_tensor(ids, dtype=torch.int64)
            pids[pids > -1] += 1
            pids[pids == -1] = 5555
            targets.append(
                {
                    "file_name": fname,
                    "image_id": imgid,
                    "boxes": bboxes,
                    "labels": pids,
                }
            )
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        detections = self.transform.postprocess(detections, images.image_sizes, org_hws)

        losses = {}
        # apply loss weights
        losses["loss_rpn_reg"] = self.lw_rpn_reg * proposal_losses["loss_rpn_box_reg"]
        losses["loss_rpn_cls"] = self.lw_rpn_cls * proposal_losses["loss_objectness"]
        losses["loss_bbox"] = self.lw_rcnn_reg * detector_losses["loss_box_reg"]
        losses["loss_ce"] = self.lw_rcnn_cls * detector_losses["loss_detection"]
        losses["loss_oim"] = self.lw_reid * detector_losses["loss_reid"]
        return losses


class HOIMRoIHeads(RoIHeads):
    def __init__(self, embedding_head, reid_loss, *args, **kwargs):
        super(HOIMRoIHeads, self).__init__(*args, **kwargs)
        self.embedding_head = embedding_head
        self.reid_loss = reid_loss

    @property
    def feat_head(self):  # this name is better
        return self.box_head

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t[
                    "boxes"
                ].dtype.is_floating_point, "target boxes must of float type"
                assert (
                    t["labels"].dtype == torch.int64
                ), "target labels must of int64 type"

        labels = None
        if self.training:
            (
                proposals,
                matched_idxs,
                labels,
                regression_targets,
            ) = self.select_training_samples(proposals, targets)

        roi_pooled_features = self.box_roi_pool(features, proposals, image_shapes)
        rcnn_features = self.feat_head(roi_pooled_features)
        box_regression = self.box_predictor(rcnn_features["feat_res5"])
        embeddings_ = self.embedding_head(rcnn_features)
        class_score, loss_detection, loss_reid = self.reid_loss(embeddings_, labels)

        result, losses = [], {}
        if self.training:
            det_labels = [y.clamp(0, 1) for y in labels]
            loss_box_reg = coord_regression_loss(
                class_score, box_regression, det_labels, regression_targets
            )

            losses = dict(
                loss_detection=loss_detection,
                loss_box_reg=loss_box_reg,
                loss_reid=loss_reid,
            )
        else:
            boxes, scores, embeddings, labels = self.postprocess_detections(
                class_score, box_regression, embeddings_, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        embeddings=embeddings[i],
                    )
                )
        # Mask and Keypoint losses are deleted deleted
        return result, losses

    def postprocess_detections(
        self, pred_scores, box_regression, embeddings_, proposals, image_shapes
    ):
        device = pred_scores.device
        num_classes = pred_scores.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings_.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(
            pred_boxes, pred_scores, pred_embeddings, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            # embeddings are already personized.

            # batch everything, by making every class prediction be a separate
            # instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels


class CoordRegressor(nn.Module):
    """
    bounding box regression layers, without classification layer.
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
                           default = 2 for pedestrian detection
    """

    def __init__(self, in_channels, num_classes=2, RCNN_bbox_bn=True):
        super(CoordRegressor, self).__init__()
        if RCNN_bbox_bn:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes), nn.BatchNorm1d(4 * num_classes)
            )
        else:
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.cls_score = nn.Linear(in_channels, num_classes)  # useless here.

        init.normal_(self.bbox_pred[0].weight, std=0.01)
        init.normal_(self.bbox_pred[1].weight, std=0.01)
        init.constant_(self.bbox_pred[0].bias, 0)
        init.constant_(self.bbox_pred[1].bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def coord_regression_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.
    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()
    return box_loss


class ReIDEmbeddingProj(nn.Module):
    def __init__(self, featmap_names=["feat_res5"], in_channels=[2048], dim=256):
        super(ReIDEmbeddingProj, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = int(dim)

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_chennel, indv_dim in zip(
            self.featmap_names, self.in_channels, indv_dims
        ):
            proj = nn.Sequential(
                nn.Linear(in_chennel, indv_dim), nn.BatchNorm1d(indv_dim)
            )
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
        """
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            v = self._flatten_fc_input(v)
            return F.normalize(self.projectors[k](v))
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))
            return F.normalize(torch.cat(outputs, dim=1))

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x  # ndim = 2, (N, d)

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [int(self.dim / parts)] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp
