from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torchvision.ops import MultiScaleRoIAlign, roi_align
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.rpn import (
    RegionProposalNetwork,
    concat_box_prediction_layers,
)

from psd2.org_model_lib.nae.model.faster_rcnn_norm_aware import (
    FasterRCNN_NormAware,
    CoordRegressor,
    NormAwareRoiHeads,
    NormAwareEmbeddingProj,
)
from psd2.org_model_lib.nae.loss import OIMLossSMR
from psd2.org_model_lib.nae.model.resnet_backbone import resnet_backbone
from ..build import META_ARCH_REGISTRY
import logging
from psd2.structures.nested_tensor import nested_collate_fn
import os

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class NAE_M(FasterRCNN_NormAware):
    def __init__(self, cfg):
        backbone, conv_head = resnet_backbone("resnet50", True)
        coord_fc = CoordRegressor(2048, num_classes=2, RCNN_bbox_bn=True)

        embedding_head = NormAwareEmbeddingProj(
            featmap_names=["feat_res4", "feat_res5"], in_channels=[1024, 2048], dim=256
        )
        search_cfg = cfg.MODEL.SEARCH
        rpn_cfg = search_cfg.NAE.RPN
        box_cfg = search_cfg.NAE.BOX
        super().__init__(
            backbone,
            feat_head=conv_head,
            box_predictor=coord_fc,
            embedding_head=embedding_head,
            num_pids=search_cfg.OIM.LUT_SIZE,
            num_cq_size=search_cfg.OIM.CQ_SIZE,
            anchor_scales=(tuple(rpn_cfg.ANCHOR_SCALES),),
            anchor_ratios=(tuple(rpn_cfg.ANCHOR_RATIOS),),
            rpn_pre_nms_top_n_train=rpn_cfg.RPN_PRE_NMS_TOP_N,
            rpn_pre_nms_top_n_test=rpn_cfg.RPN_PRE_NMS_TOP_N_TEST,
            rpn_post_nms_top_n_train=rpn_cfg.RPN_POST_NMS_TOP_N,
            rpn_post_nms_top_n_test=rpn_cfg.RPN_POST_NMS_TOP_N_TEST,
            rpn_nms_thresh=rpn_cfg.RPN_NMS_THRESH,
            rpn_fg_iou_thresh=rpn_cfg.RPN_POSITIVE_OVERLAP,
            rpn_bg_iou_thresh=rpn_cfg.RPN_NEGATIVE_OVERLAP,
            rpn_batch_size_per_image=rpn_cfg.RPN_BATCH_SIZE,
            rpn_positive_fraction=rpn_cfg.RPN_FG_FRACTION,
            rcnn_bbox_bn=True,
            box_score_thresh=box_cfg.FG_THRESH,
            box_nms_thresh=box_cfg.NMS_TEST,
            box_detections_per_img=rpn_cfg.RPN_POST_NMS_TOP_N_TEST,
            box_fg_iou_thresh=box_cfg.BG_THRESH_HI,
            box_bg_iou_thresh=box_cfg.BG_THRESH_LO,
            box_batch_size_per_image=box_cfg.RCNN_BATCH_SIZE,
            box_positive_fraction=box_cfg.FG_FRACTION,
            bbox_reg_weights=box_cfg.BOX_REGRESSION_WEIGHTS,
        )
        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
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
        trained_model_path = cfg.MODEL.SEARCH.NAE.WEIGHTS
        self._resume_from_ckpt(trained_model_path)

    def _resume_from_ckpt(self, ckpt_path, optimizer=None, lr_scheduler=None):
        if not os.path.exists(ckpt_path) or len(ckpt_path) == 0:
            logger.info(f"No checkpoint found at {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["model"], strict=False)
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        logger.info(f"loaded checkpoint {ckpt_path}")
        logger.info(f"model was trained for {ckpt['epoch']} epochs")

    @property
    def device(self):
        return self.pixel_mean.device

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
            if isinstance(rcnn_features, torch.Tensor):
                rcnn_features = OrderedDict([("feat_res5", rcnn_features)])
            embeddings, norms = self.roi_heads.embedding_head(rcnn_features)
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


@META_ARCH_REGISTRY.register()
class NAE(FasterRCNN_NormAware):
    def __init__(self, cfg):
        featmap_names = ["feat_res5"]
        in_channels = [2048]
        return_res4 = False
        backbone, conv_head = resnet_backbone(
            "resnet50", True, return_res4=return_res4, GAP=False
        )
        pool_module = MultiScaleRoIAlign
        roi_pooler = pool_module(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )

        coord_fc = CoordRegressor(2048, num_classes=2, RCNN_bbox_bn=True)

        embedding_head = PixelWiseNormAwareEmbeddingProj(
            featmap_names=featmap_names, in_channels=in_channels, dim=256
        )
        search_cfg = cfg.MODEL.SEARCH
        rpn_cfg = search_cfg.NAE.RPN
        box_cfg = search_cfg.NAE.BOX
        super().__init__(
            backbone,
            num_pids=search_cfg.OIM.LUT_SIZE,
            num_cq_size=search_cfg.OIM.CQ_SIZE,
            rpn_pre_nms_top_n_train=rpn_cfg.RPN_PRE_NMS_TOP_N,
            rpn_pre_nms_top_n_test=rpn_cfg.RPN_PRE_NMS_TOP_N_TEST,
            rpn_post_nms_top_n_train=rpn_cfg.RPN_POST_NMS_TOP_N,
            rpn_post_nms_top_n_test=rpn_cfg.RPN_POST_NMS_TOP_N_TEST,
            rpn_nms_thresh=rpn_cfg.RPN_NMS_THRESH,
            rpn_fg_iou_thresh=rpn_cfg.RPN_POSITIVE_OVERLAP,
            rpn_bg_iou_thresh=rpn_cfg.RPN_NEGATIVE_OVERLAP,
            rpn_batch_size_per_image=rpn_cfg.RPN_BATCH_SIZE,
            rpn_positive_fraction=rpn_cfg.RPN_FG_FRACTION,
            rcnn_bbox_bn=True,
            box_roi_pool=roi_pooler,
            feat_head=conv_head,
            box_predictor=coord_fc,
            box_score_thresh=box_cfg.FG_THRESH,
            box_nms_thresh=box_cfg.NMS_TEST,
            box_detections_per_img=rpn_cfg.RPN_POST_NMS_TOP_N_TEST,
            box_fg_iou_thresh=box_cfg.BG_THRESH_HI,
            box_bg_iou_thresh=box_cfg.BG_THRESH_LO,
            box_batch_size_per_image=box_cfg.RCNN_BATCH_SIZE,
            box_positive_fraction=box_cfg.FG_FRACTION,
            bbox_reg_weights=box_cfg.BOX_REGRESSION_WEIGHTS,
            embedding_head=embedding_head,
            reid_loss=None,
        )
        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
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
        trained_model_path = cfg.MODEL.SEARCH.NAE.WEIGHTS
        self._resume_from_ckpt(trained_model_path)

    def _resume_from_ckpt(self, ckpt_path, optimizer=None, lr_scheduler=None):
        if not os.path.exists(ckpt_path) or len(ckpt_path) == 0:
            logger.info(f"No checkpoint found at {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["model"], strict=False)
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        logger.info(f"loaded checkpoint {ckpt_path}")
        logger.info(f"model was trained for {ckpt['epoch']} epochs")

    @property
    def device(self):
        return self.pixel_mean.device

    def _set_roi_heads(self, *args):
        return PixelWiseNormAwareRoiHeads(*args)

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
        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([("feat_res5", rcnn_features)])
        embeddings, class_logits = self.roi_heads.embedding_head(rcnn_features)
        spatial_attention = torch.sigmoid(class_logits)
        embeddings = F.adaptive_avg_pool2d(embeddings * spatial_attention, 1).flatten(
            start_dim=1
        )
        return embeddings.split(1, 0)

    def ex_feat_by_img_crop(self, images, targets):
        assert len(images) == 1, "Only support batch_size 1 in this mode"

        images, targets = self.transform(images, targets)
        x1, y1, x2, y2 = map(lambda x: int(round(x)), targets[0]["boxes"][0].tolist())
        input_tensor = images.tensors[:, :, y1 : y2 + 1, x1 : x2 + 1]
        features = self.backbone(input_tensor)
        features = features.values()[0]
        rcnn_features = self.roi_heads.feat_head(features)
        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([("feat_res5", rcnn_features)])
        embeddings, class_logits = self.roi_heads.embedding_head(rcnn_features)
        spatial_attention = torch.sigmoid(class_logits)
        embeddings = F.adaptive_avg_pool2d(embeddings * spatial_attention, 1).flatten(
            start_dim=1
        )
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
            if isinstance(rcnn_features, torch.Tensor):
                rcnn_features = OrderedDict([("feat_res5", rcnn_features)])
            embeddings, class_logits = self.roi_heads.embedding_head(rcnn_features)
            spatial_attention = torch.sigmoid(class_logits)
            embeddings = F.adaptive_avg_pool2d(
                embeddings * spatial_attention, 1
            ).flatten(start_dim=1)
            p_feats = embeddings.split(1, 0)[0].cpu()
            result_list = input_list.copy()
            for bi, feat in enumerate(p_feats):
                result_list[bi]["query"]["feat"] = feat
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


class PixelWiseNormAwareRoiHeads(NormAwareRoiHeads):
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
                """if self.has_keypoint:
                    assert (
                        t["keypoints"].dtype == torch.float32
                    ), "target keypoints must of float type"""

        if self.training:
            (
                proposals,
                matched_idxs,
                labels,
                regression_targets,
                matched_gt_boxes,
            ) = self.select_training_samples(proposals, targets)

        rcnn_features = self.feat_head(
            self.box_roi_pool(features, proposals, image_shapes)
        )  # size = (N, C, 7, 7)

        box_regression = self.box_predictor(rcnn_features["feat_res5"])
        embeddings_, class_logits = self.embedding_head(
            rcnn_features
        )  # size = (N, d, 7, 7) and (N, 1, 7, 7)
        spatial_attention = torch.sigmoid(class_logits)
        embeddings_ = F.adaptive_avg_pool2d(embeddings_ * spatial_attention, 1).flatten(
            start_dim=1
        )  # size = (N, d)

        result, losses = [], {}
        if self.training:
            # Generate pixel-wise label
            spatial_labels = self.grid_wise_label_gen(
                matched_gt_boxes, proposals, size=rcnn_features["feat_res5"].shape[2:]
            )
            det_labels = [y.clamp(0, 1) for y in labels]
            loss_detection, loss_box_reg = spatial_norm_aware_rcnn_loss(
                class_logits,
                box_regression,
                spatial_labels,
                det_labels,
                regression_targets,
            )
            loss_reid = self.reid_loss(embeddings_, labels)

            losses = dict(
                loss_detection=loss_detection,
                loss_box_reg=loss_box_reg,
                loss_reid=loss_reid,
            )
        else:
            boxes, scores, embeddings, labels = self.postprocess_detections(
                class_logits, box_regression, embeddings_, proposals, image_shapes
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
        # Mask and Keypoint losses are deleted
        return result, losses

    def postprocess_detections(
        self, class_logits, box_regression, embeddings_, proposals, image_shapes
    ):
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = torch.sigmoid(class_logits).mean(dim=(2, 3))
        pred_scores /= pred_scores.max()

        embeddings_ = F.normalize(embeddings_)
        embeddings_ = embeddings_ * pred_scores.view(-1, 1)

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
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)
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

    @staticmethod
    def grid_wise_label_gen(matched_gt_boxes, proposals, size=(7, 7)):
        cat_gt = torch.cat(matched_gt_boxes, dim=0)
        cat_p = torch.cat(proposals, dim=0)
        num_proposals = cat_gt.size(0)
        grid_wise_labels = torch.zeros(num_proposals, 1, *size).to(proposals[0].device)
        for i in range(num_proposals):
            width = (cat_p[i][2] - cat_p[i][0]).ceil().long().item()
            height = (cat_p[i][3] - cat_p[i][1]).ceil().long().item()
            if not (width > 0 and height > 0):
                continue  # invalid proposal
            tmp = torch.zeros(1, 1, height, width)  # same 'size' as proposal
            x1 = (
                (torch.max(cat_p[i][0], cat_gt[i][0]) - cat_p[i][0])
                .floor()
                .long()
                .item()
            )
            y1 = (
                (torch.max(cat_p[i][1], cat_gt[i][1]) - cat_p[i][1])
                .floor()
                .long()
                .item()
            )
            x2 = (
                (torch.min(cat_p[i][2], cat_gt[i][2]) - cat_p[i][0])
                .ceil()
                .long()
                .item()
            )
            y2 = (
                (torch.min(cat_p[i][3], cat_gt[i][3]) - cat_p[i][1])
                .ceil()
                .long()
                .item()
            )
            tmp[0, 0, y1 : y2 + 1, x1 : x2 + 1] = 1.0
            grid_wise_labels[i] = F.interpolate(
                tmp, size=size, mode="bilinear", align_corners=False
            )
        return grid_wise_labels

    def select_training_samples(self, proposals, targets):
        """
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/roi_heads.py#L445
        """
        self.check_targets(targets)
        dtype = proposals[0].dtype
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels
        )
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, matched_gt_boxes


class PixelWiseNormAwareEmbeddingProj(NormAwareEmbeddingProj):
    """
    Current Version:
        1. conv kernel shared by all locations
        2. average pool instead of resize/concat, so the output dim = embedding dim
    TODO: Version 2: don't share conv kernels.

    """

    def __init__(self, *args, **kwargs):
        super(PixelWiseNormAwareEmbeddingProj, self).__init__(*args, **kwargs)
        self.rescaler = nn.BatchNorm2d(1, affine=True)
        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_chennel, indv_dim in zip(
            self.featmap_names, self.in_channels, indv_dims
        ):
            proj = nn.Sequential(
                nn.Conv2d(in_chennel, int(indv_dim), kernel_size=1),
                nn.BatchNorm2d(int(indv_dim)),
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
            tensor of size (BatchSize, dim, gird_size[0], grid_size[1]), L2 normalized embeddings.
            tensor of size (BatchSize, 1, gird_size[0], grid_size[1]) rescaled norm of embeddings, as class_logits.
        """
        if len(featmaps) == 1:
            k, v = list(featmaps.items())[0]
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.clamp(min=1e-12)
            norms = self.rescaler(norms)
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                outputs.append(self.projectors[k](v))
            outputs[0] = F.interpolate(
                outputs[0],
                size=outputs[1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.clamp(min=1e-12)
            norms = self.rescaler(norms)
            return embeddings, norms

    @property
    def rescaler_weight(self):
        return self.rescaler.weight.item()


def spatial_norm_aware_rcnn_loss(
    class_logits,
    box_regression,
    spatial_labels,
    labels,
    regression_targets,
    focal=False,
    alpha_d=0.25,
    gamma_d=2.0,
):
    """
    Computes the loss for grid/pixel-wise Norm-Aware R-CNN.
    Arguments:
        class_logits (Tensor), size = (N, 1, h, w)
        box_regression (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.binary_cross_entropy_with_logits(
        class_logits, spatial_labels, reduction="none"
    )
    if focal:
        pt = torch.exp(-classification_loss)
        classification_loss = alpha_d * (1 - pt) ** gamma_d * classification_loss

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N = class_logits.size(0)
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss.mean(), box_loss


def load_NAE_weights(args, model):
    if args.dataset == "CUHK-SYSU":
        model_path = "logs/cuhk-sysu/checkpoint.pth"
    elif args.dataset == "PRW":
        model_path = "logs/prw/checkpoint.pth"
    checkpoint = torch.load(model_path)

    state_dict = checkpoint["model"]
    state_dict["roi_heads.embedding_head.projectors.feat_res4.0.weight"] = state_dict[
        "roi_heads.embedding_head.projectors.feat_res4.0.weight"
    ].view(128, 1024, 1, 1)
    state_dict["roi_heads.embedding_head.projectors.feat_res5.0.weight"] = state_dict[
        "roi_heads.embedding_head.projectors.feat_res5.0.weight"
    ].view(128, 2048, 1, 1)

    model.load_state_dict(state_dict)
    print(hue.good("NAE pre-trained weights loaded."))
    return model
