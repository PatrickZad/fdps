from copy import deepcopy
from turtle import forward, pos

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    RPNHead,
)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from ..build import META_ARCH_REGISTRY
import logging
from psd2.structures.nested_tensor import nested_collate_fn_idvi as nested_collate_fn
import torchvision.transforms.functional as tvF
from PIL import Image
import os
from psd2.org_model_lib.coat.models import resnet as coat_resnet
from psd2.org_model_lib.coat.models import transformer as coat_trans
from psd2.org_model_lib.coat.loss import oim as coat_oim
from torchvision.models.detection import _utils as det_utils
from psd2.utils.visualizer import Visualizer, mlvl_pca_feat
from psd2.utils.events import get_event_storage
from psd2.utils import comm
from torchvision.ops import boxes as box_ops
from psd2.modeling.transformer.coat_trans import (
    TransformerHeadL2P,
    TransformerHeadL2PSimBox,
    TransformerHeadL2PSimReid,
    TransformerHeadSimBox,
    TransformerHeadSimReid,
)

logger = logging.getLogger(__name__)


def get_mem_use():
    return torch.cuda.max_memory_allocated() / 1024.0 / 1024.0


@META_ARCH_REGISTRY.register()
class BASE_COAT(nn.Module):
    def __init__(self, cfg):
        super(BASE_COAT, self).__init__()

        backbone, res5_head = coat_resnet.build_resnet(name="resnet50", pretrained=True)
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.DETECTOR.MODEL.RPN.PRE_NMS_TOPN_TRAIN,
            testing=cfg.DETECTOR.MODEL.RPN.PRE_NMS_TOPN_TEST,
        )
        post_nms_top_n = dict(
            training=cfg.DETECTOR.MODEL.RPN.POST_NMS_TOPN_TRAIN,
            testing=cfg.DETECTOR.MODEL.RPN.POST_NMS_TOPN_TEST,
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.DETECTOR.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.DETECTOR.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.DETECTOR.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.DETECTOR.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.DETECTOR.MODEL.RPN.NMS_THRESH,
        )
        if cfg.DETECTOR.MODEL.ROI_HEAD.USE_RES5:
            box_head = res5_head
        else:
            # trans head
            box_head = coat_trans.TransformerHead(
                cfg=cfg.REID_HEAD,
                kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
            )
        reid_head = coat_trans.TransformerHead(
            cfg=cfg.REID_HEAD,
            kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
        )

        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        box_predictor = BBoxRegressor(
            2048, num_classes=2, bn_neck=cfg.DETECTOR.MODEL.ROI_HEAD.BN_NECK
        )
        roi_heads = TransROIHeads(
            cfg=cfg,
            # Cascade Transformer Head
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.DETECTOR.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.DETECTOR.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.DETECTOR.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )

        if self.training:
            transform = GeneralizedRCNNTransform(
                min_size=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )
        else:
            transform = GeneralizedRCNNTransform(
                min_size=cfg.INPUT.MIN_SIZE_TEST,
                max_size=cfg.INPUT.MAX_SIZE_TEST,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )  # only to use post-process

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        lw_cfg = cfg.DETECTOR.LOSS.LOSS_WEIGHTS
        self.lw_rpn_reg = lw_cfg.LW_RPN_REG
        self.lw_rpn_cls = lw_cfg.LW_RPN_CLS
        self.lw_rcnn_reg = lw_cfg.LW_RCNN_REG
        self.lw_rcnn_cls = lw_cfg.LW_RCNN_CLS
        self.lw_rcnn_reid = cfg.REID_HEAD.LOSS.LOSS_WEIGHTS.OIM

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        self.vis_period = cfg.VIS_PERIOD

    @property
    def device(self):
        return self.pixel_mean.device

    def preproxess_input(self, input_list):
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
            input_batches = self.preproxess_input([qd["query"] for qd in input_list])
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
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(
                features, boxes, images.image_sizes
            )
            box_features_2nd = self.roi_heads.box_head_2nd(box_features)
            if isinstance(box_features_2nd, tuple):
                box_features_2nd = box_features_2nd[0]
            embeddings_2nd, _ = self.roi_heads.embedding_head_2nd(
                {"after_trans": box_features_2nd["after_trans"]}
            )
            embeddings = embeddings_2nd.cpu().split(1, 0)
            result_list = input_list.copy()
            for bi, feat in enumerate(embeddings):
                result_list[bi]["query"]["feat"] = feat.view(-1)
            return result_list
        else:
            # gallery
            input_batches = self.preproxess_input(input_list)
            images = input_batches[0]
            targets = None
            org_hws = input_batches[6]
            features = self.backbone(images.tensors)
            boxes, _ = self.rpn(images, features, targets)
            detections = self.roi_heads(features, boxes, images.image_sizes, targets)[0]
            detections = self.transform.postprocess(
                detections, images.image_sizes, org_hws
            )
            return_result = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for det in detections:
                """return_result["pred_boxes"].append(
                    box_xyxy_to_cxcywh(det["boxes"].view(-1, 4))
                )"""
                return_result["pred_boxes"].append(det["boxes"].view(-1, 4))
                return_result["pred_scores"].append(det["scores"].view(-1, 1))
                return_result["reid_feats"].append(det["embeddings"].view(-1, 256))
            # vis_inf(input_batches, return_result, "outputs/test_debug", 0.0)
            return return_result

    def forward(self, input_list):
        if not self.training:
            return self.inference(input_list)
        input_batches = self.preproxess_input(input_list)
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
        boxes, rpn_losses = self.rpn(images, features, targets)

        (
            result,
            rcnn_losses,
            feats_reid_2nd,
            targets_reid_2nd,
        ) = self.roi_heads(features, boxes, images.image_sizes, targets)

        # rename rpn losses to be consistent with detection losses
        rpn_losses["loss_rpn_reg"] = rpn_losses.pop("loss_rpn_box_reg")
        rpn_losses["loss_rpn_cls"] = rpn_losses.pop("loss_objectness")

        losses = {}
        losses.update(rcnn_losses)
        losses.update(rpn_losses)

        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_rcnn_reg_1st"] *= self.lw_rcnn_reg
        losses["loss_rcnn_cls_1st"] *= self.lw_rcnn_cls
        losses["loss_rcnn_reg_2nd"] *= self.lw_rcnn_reg
        losses["loss_rcnn_cls_2nd"] *= self.lw_rcnn_cls
        losses["loss_rcnn_reid_2nd"] *= self.lw_rcnn_reid
        for lv in losses.values():
            if torch.isnan(lv).sum() > 0:
                msg = "nan at " + " ".join(
                    [input_dict["file_name"] for input_dict in input_list]
                )
                print(msg)
                logger.error(msg)
                break
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(
                    (images, targets),
                    features[list(features.keys())[-1]],
                    result,
                )

        return losses

    @torch.no_grad()
    def visualize_training(self, batched_inputs, featmap, batched_dets):
        """
        Args:
            batched_inputs:
                [imgs nested tensor, img_gt_instances]
            featmap: a batched images feature map tensor
            featmap: (multi level) image feature map(s)
            batched_dets:
                {
                    "pred_logits": batched person scores B x N x 1,
                    "pred_boxes": batched xyxy bboxes in [0,1],
                    "reid_feat": reid feature,
                    "assign_ids": ids assigned to each reid feature
                    "aux_outputs": optional,
                ]
            vals: scalars to be visualized
            thred: cls score threshold
        """
        import torchvision.transforms.functional as tvF
        import numpy as np

        COLORS = ["r", "g", "b", "y", "c", "m"]
        T_COLORS_BG = {
            "r": "white",
            "g": "white",
            "b": "white",
            "y": "black",
            "c": "black",
            "m": "white",
        }
        storage = get_event_storage()
        trans_t2img_t = lambda t: t.detach().cpu() * self.pixel_std.cpu().view(
            -1, 1, 1
        ) + self.pixel_mean.cpu().view(-1, 1, 1)
        img_t2rgb = lambda t: (t.permute(1, 2, 0) * 255).numpy()
        samples = batched_inputs[0]
        annos = []
        for inst in batched_inputs[1]:
            trans_id = inst["labels"]
            trans_id[trans_id == 5555] = -1
            trans_id[trans_id > 0] -= 1
            annos.append(
                {
                    "file_name": inst["file_name"],
                    "image_id": inst["image_id"],
                    "boxes": inst["boxes"],  # xyxy abs
                    "ids": trans_id,
                }
            )
        bs = len(annos)
        level_pcas = mlvl_pca_feat(featmap[None])
        for bi in range(bs):
            img_norm_t = samples.tensors[bi].cpu()
            img_t = trans_t2img_t(img_norm_t)
            img_rgb = img_t2rgb(img_t)
            visualize_org = Visualizer(img_rgb.copy())
            boxes = annos[bi]["boxes"].cpu()
            ids = annos[bi]["ids"].tolist()
            for i in range(boxes.shape[0]):
                visualize_org.draw_box(boxes[i])
                id_pos = boxes[i, :2]
                visualize_org.draw_text(
                    str(ids[i]), id_pos, horizontal_alignment="left", color="w"
                )

            rgb_org_ann = visualize_org.get_output().get_image()
            t_org_ann = tvF.to_tensor(rgb_org_ann)
            storage.put_image("img_{}/gt".format(bi), t_org_ann)

            visualize_run = Visualizer(img_rgb.copy())
            boxes = batched_dets[bi]["boxes"].detach().cpu()
            for i, bx in enumerate(boxes):
                b_clr = COLORS[i % len(COLORS)]
                t_clr = T_COLORS_BG[b_clr]
                pid = batched_dets[bi]["labels"][i].item()
                if pid == 0:
                    continue
                if pid == 5555:
                    pid = -1
                else:
                    pid -= 1
                visualize_run.draw_box(bx, edge_color=b_clr)
                visualize_run.draw_text(
                    str(pid),
                    boxes[i][:2],
                    horizontal_alignment="left",
                    color=t_clr,
                    bg_color=b_clr,
                )
            rgb_run_vis = visualize_run.get_output().get_image()
            t_run_vis = tvF.to_tensor(rgb_run_vis)
            storage.put_image("img_{}/det".format(bi), t_run_vis)
            max_h, w = 0, 0
            level_pca_feats = []
            for feat in level_pcas:
                bi_pca = feat[bi]
                level_pca_feats.append(bi_pca)
                if bi_pca.shape[-2] > max_h:
                    max_h = bi_pca.shape[-2]
                w += bi_pca.shape[-1]
            bg = torch.zeros(3, max_h, w, dtype=torch.float32)
            next_w = 0
            for feat in level_pcas:
                bg[
                    :, : feat[bi].shape[-2], next_w : next_w + feat[bi].shape[-1]
                ] = feat[bi]
                next_w += feat[bi].shape[-1]
            storage.put_image("img_{}/feat".format(bi), bg / 255)


class TransROIHeads(RoIHeads):
    """
    https://github.com/pytorch/vision/blob/master/torchvision/models/detection/roi_heads.py
    """

    def __init__(self, cfg, faster_rcnn_predictor, reid_head, *args, **kwargs):
        super(TransROIHeads, self).__init__(*args, **kwargs)

        # ROI head
        self.use_diff_thresh = cfg.DETECTOR.MODEL.ROI_HEAD.USE_DIFF_THRESH
        self.nms_thresh_1st = cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST
        self.nms_thresh_2nd = cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST_2ND
        self.fg_iou_thresh_1st = cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN
        self.bg_iou_thresh_1st = cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN
        self.fg_iou_thresh_2nd = cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN_2ND
        self.bg_iou_thresh_2nd = cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_2ND

        # Regression head
        self.box_predictor_1st = faster_rcnn_predictor
        self.box_predictor_2nd = self.box_predictor

        # Transformer head
        self.box_head_1st = self.box_head
        self.box_head_2nd = reid_head

        # Feature embedding
        ehead_cfg = cfg.REID_HEAD.EMBEDDING_HEAD
        embedding_dim = ehead_cfg.EMBEDDING_DIM
        ehead_infeats = ehead_cfg.IN_FEAT
        feat_channels = {
            "before_trans": 1024,
            "after_trans": 2048,
            "feat_res4": 1024,
            "feat_res5": 2048,
        }
        self.embedding_head_2nd = NormAwareEmbedding(
            featmap_names=ehead_infeats,
            in_channels=[feat_channels[fn] for fn in ehead_infeats],
            dim=embedding_dim,
        )
        self.nae_det = cfg.REID_HEAD.EMBEDDING_HEAD.DET_SUP
        # OIM
        oim_cfg = cfg.REID_HEAD.LOSS.OIM
        num_pids = oim_cfg.LUT_SIZE
        num_cq_size = oim_cfg.CQ_SIZE
        oim_momentum = oim_cfg.OIM_MOMENTUM
        oim_scalar = oim_cfg.OIM_SCALAR
        self.reid_loss_2nd = coat_oim.OIMLoss(
            embedding_dim, num_pids, num_cq_size, oim_momentum, oim_scalar
        )

        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections

    def forward(self, features, boxes, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            boxes (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        gt_det_2nd = None
        feats_reid_2nd = None
        targets_reid_2nd = None

        if self.training:
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_1st,
                    self.bg_iou_thresh_1st,
                    allow_low_quality_matches=False,
                )
            (
                boxes,
                _,
                box_pid_labels_1st,
                box_reg_targets_1st,
            ) = self.select_training_samples(
                boxes, targets
            )  # 2000 input from rpn

        # ------------------- The first stage ------------------ # trans feat head + faster_rcnn_predictor
        box_features_1st = self.box_roi_pool(features, boxes, image_shapes)
        box_features_1st = self.box_head_1st(box_features_1st)
        if "after_trans" in box_features_1st:
            out_name = "after_trans"
        else:
            out_name = "feat_res5"
        box_cls_scores_1st, box_regs_1st = self.box_predictor_1st(
            box_features_1st[out_name]
        )

        if self.training:
            boxes = self.get_boxes(box_regs_1st, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_2nd,
                    self.bg_iou_thresh_2nd,
                    allow_low_quality_matches=False,
                )
            (
                boxes,
                _,
                box_pid_labels_2nd,
                box_reg_targets_2nd,
            ) = self.select_training_samples(boxes, targets)
        else:
            orig_thresh = self.nms_thresh  # 0.4
            self.nms_thresh = self.nms_thresh_1st
            boxes, scores, _ = self.postprocess_proposals(
                box_cls_scores_1st, box_regs_1st, boxes, image_shapes
            )
            self.nms_thresh = orig_thresh

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes_i = torch.zeros(0, 4)
            labels_i = torch.zeros(0)
            scores_i = torch.zeros(0)
            embeddings_i = torch.zeros(0, 256)
            return [
                dict(boxes=boxes_i, scores=scores_i, embeddings=embeddings_i)
                for _ in range(len(boxes))
            ], []

        # --------------------- The second stage -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features = self.box_head_2nd(box_features)
        if "after_trans" in box_features:
            out_name = "after_trans"
        else:
            out_name = "feat_res5"
        box_regs_2nd = self.box_predictor_2nd(box_features[out_name])
        box_embeddings_2nd, box_cls_scores_2nd = self.embedding_head_2nd(
            {out_name: box_features[out_name]}
        )
        if box_cls_scores_2nd.dim() == 0:
            box_cls_scores_2nd = box_cls_scores_2nd.unsqueeze(0)

        result, losses = [], {}
        if self.training:
            box_labels_1st = [y.clamp(0, 1) for y in box_pid_labels_1st]
            box_labels_2nd = [y.clamp(0, 1) for y in box_pid_labels_2nd]
            if self.nae_det:
                losses = detection_losses(
                    box_cls_scores_1st,
                    box_regs_1st,
                    box_labels_1st,
                    box_reg_targets_1st,
                    box_cls_scores_2nd,
                    box_regs_2nd,
                    box_labels_2nd,
                    box_reg_targets_2nd,
                )
            else:
                losses = detection_losses(
                    box_cls_scores_1st,
                    box_regs_1st,
                    box_labels_1st,
                    box_reg_targets_1st,
                    box_cls_scores_2nd,
                    None,
                    None,
                    None,
                )

            (
                loss_rcnn_reid_2nd,
                feats_reid_2nd,
                targets_reid_2nd,
                valid_inds,
            ) = self.reid_loss_2nd(box_embeddings_2nd, box_pid_labels_2nd)
            losses.update(loss_rcnn_reid_2nd=loss_rcnn_reid_2nd)
            # for vis only
            valid_inds_per_img = torch.split(
                valid_inds, [y.shape[0] for y in box_pid_labels_2nd]
            )
            valid_boxes = [b[vi] for b, vi in zip(boxes, valid_inds_per_img)]
            pid_labels_per_img = [
                pid[vi] for pid, vi in zip(box_pid_labels_2nd, valid_inds_per_img)
            ]
            for i in range(len(boxes)):
                result.append(
                    dict(
                        boxes=valid_boxes[i],
                        labels=pid_labels_per_img[i],
                    )
                )

        else:
            if self.nae_det:
                boxes, scores, embeddings_2nd, labels = self.postprocess_boxes(
                    box_cls_scores_2nd,
                    box_regs_2nd,
                    box_embeddings_2nd,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=None,
                    cws=False,
                )
            else:
                embeddings_2nd = torch.split(
                    box_embeddings_2nd, [bx.shape[0] for bx in boxes]
                )
            num_images = len(boxes)
            for i in range(num_images):
                embeddings = embeddings_2nd[i]
                result.append(
                    dict(
                        boxes=boxes[i],
                        scores=scores[i],
                        embeddings=embeddings,
                    )
                )
                num_images = len(boxes)
                for i in range(num_images):
                    embeddings = embeddings_2nd[i]
                    result.append(
                        dict(
                            boxes=boxes[i],
                            scores=scores[i],
                            embeddings=embeddings,
                        )
                    )

        return (
            result,
            losses,
            feats_reid_2nd,
            targets_reid_2nd,
        )

    def get_boxes(self, box_regression, proposals, image_shapes):
        """
        Get boxes from proposals.
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # remove predictions with the background label
            boxes = boxes[:, 1:].reshape(-1, 4)
            all_boxes.append(boxes)

        return all_boxes

    def postprocess_boxes(
        self,
        class_logits,
        box_regression,
        embeddings,
        proposals,
        image_shapes,
        fcs=None,
        gt_det=None,
        cws=True,
    ):
        """
        Similar to RoIHeads.postprocess_detections, but can handle embeddings and implement
        First Classification Score (FCS).
        """
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        if fcs is not None:
            # Fist Classification Score (FCS)
            pred_scores = torch.cat(fcs, dim=0)  # fcs[0]
        else:
            pred_scores = torch.sigmoid(class_logits)
        if cws:
            # Confidence Weighted Similarity (CWS)
            embeddings = embeddings * pred_scores.view(-1, 1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings.split(boxes_per_image, 0)

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

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head_2nd.dim)

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

            if gt_det is not None:
                # include GT into the detection results
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

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


class TransROIHeadsSim(TransROIHeads):
    def __init__(self, cfg, reid_head, *args, **kwargs):
        super(TransROIHeads, self).__init__(*args, **kwargs)

        # ROI head
        self.use_diff_thresh = cfg.DETECTOR.MODEL.ROI_HEAD.USE_DIFF_THRESH
        self.nms_thresh_1st = cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST
        self.nms_thresh_2nd = cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST_2ND
        self.fg_iou_thresh_1st = cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN
        self.bg_iou_thresh_1st = cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN
        self.fg_iou_thresh_2nd = cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN_2ND
        self.bg_iou_thresh_2nd = cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_2ND

        # Regression head
        self.box_predictor_1st = self.box_predictor

        # Transformer head
        self.box_head_1st = self.box_head
        self.box_head_2nd = reid_head

        # OIM
        oim_cfg = cfg.REID_HEAD.LOSS.OIM
        num_pids = oim_cfg.LUT_SIZE
        num_cq_size = oim_cfg.CQ_SIZE
        oim_momentum = oim_cfg.OIM_MOMENTUM
        oim_scalar = oim_cfg.OIM_SCALAR
        self.reid_loss_2nd = coat_oim.OIMLoss(
            cfg.REID_HEAD.DIM_MODEL, num_pids, num_cq_size, oim_momentum, oim_scalar
        )

        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections

        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections
        self.bn_neck = nn.BatchNorm1d(cfg.REID_HEAD.DIM_MODEL)

    def forward(self, features, boxes, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            boxes (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        gt_det_2nd = None
        feats_reid_2nd = None
        targets_reid_2nd = None

        if self.training:
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_1st,
                    self.bg_iou_thresh_1st,
                    allow_low_quality_matches=False,
                )
            (
                boxes,
                _,
                box_pid_labels_1st,
                box_reg_targets_1st,
            ) = self.select_training_samples(
                boxes, targets
            )  # 2000 input from rpn

        # ------------------- The first stage ------------------ # trans feat head + faster_rcnn_predictor
        box_features_1st = self.box_roi_pool(features, boxes, image_shapes)
        box_features_1st = self.box_head_1st(box_features_1st)
        if "after_trans" in box_features_1st:
            out_name = "after_trans"
        else:
            out_name = "feat_res5"
        box_cls_scores_1st, box_regs_1st = self.box_predictor_1st(
            box_features_1st[out_name]
        )

        if self.training:
            boxes = self.get_boxes(box_regs_1st, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_2nd,
                    self.bg_iou_thresh_2nd,
                    allow_low_quality_matches=False,
                )
            (
                boxes,
                _,
                box_pid_labels_2nd,
                box_reg_targets_2nd,
            ) = self.select_training_samples(boxes, targets)
        else:
            orig_thresh = self.nms_thresh  # 0.4
            self.nms_thresh = self.nms_thresh_1st
            boxes, scores, _ = self.postprocess_proposals(
                box_cls_scores_1st, box_regs_1st, boxes, image_shapes
            )
            self.nms_thresh = orig_thresh

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes = torch.zeros(0, 4)
            labels = torch.zeros(0)
            scores = torch.zeros(0)
            embeddings = torch.zeros(0, 256)
            return [
                dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)
                for _ in range(len(boxes))
            ], []

        # --------------------- The second stage -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features = self.box_head_2nd(box_features)
        if "after_trans" in box_features:
            out_name = "after_trans"
        else:
            out_name = "feat_res5"
        box_embeddings_2nd = box_features[out_name].squeeze(-1).squeeze(-1)

        # not l2 normed
        box_embeddings_2nd = self.bn_neck(box_embeddings_2nd)
        box_embeddings_2nd = F.normalize(box_embeddings_2nd, dim=1)
        # embeddings already l2 normed
        del box_features

        result, losses = [], {}
        if self.training:
            box_labels_1st = [y.clamp(0, 1) for y in box_pid_labels_1st]
            losses = detection_losses(
                box_cls_scores_1st,
                box_regs_1st,
                box_labels_1st,
                box_reg_targets_1st,
                None,
                None,
                None,
                None,
            )

            (
                loss_rcnn_reid_2nd,
                feats_reid_2nd,
                targets_reid_2nd,
                valid_inds,
            ) = self.reid_loss_2nd(box_embeddings_2nd, box_pid_labels_2nd)
            losses.update(loss_rcnn_reid_2nd=loss_rcnn_reid_2nd)
            # for vis only
            valid_inds_per_img = torch.split(
                valid_inds, [y.shape[0] for y in box_pid_labels_2nd]
            )
            valid_boxes = [b[vi] for b, vi in zip(boxes, valid_inds_per_img)]
            pid_labels_per_img = [
                pid[vi] for pid, vi in zip(box_pid_labels_2nd, valid_inds_per_img)
            ]
            for i in range(len(boxes)):
                result.append(
                    dict(
                        boxes=valid_boxes[i],
                        labels=pid_labels_per_img[i],
                    )
                )

        else:
            embeddings_2nd = torch.split(
                box_embeddings_2nd, [bx.shape[0] for bx in boxes]
            )
            num_images = len(boxes)
            for i in range(num_images):
                embeddings = embeddings_2nd[i]
                result.append(
                    dict(
                        boxes=boxes[i],
                        scores=scores[i],
                        embeddings=embeddings,
                    )
                )
                num_images = len(boxes)
                for i in range(num_images):
                    embeddings = embeddings_2nd[i]
                    result.append(
                        dict(
                            boxes=boxes[i],
                            scores=scores[i],
                            embeddings=embeddings,
                        )
                    )

        return (
            result,
            losses,
            feats_reid_2nd,
            targets_reid_2nd,
        )


class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(
        self,
        featmap_names=["feat_res4", "feat_res5"],
        in_channels=[1024, 2048],
        dim=256,
    ):
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(
            self.featmap_names, self.in_channels, indv_dims
        ):
            proj = nn.Sequential(
                nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim)
            )
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        assert len(featmaps) == len(self.featmap_names)
        if len(featmaps) == 1:
            k, v = list(featmaps.items())[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim // parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp


class BBoxRegressor(nn.Module):
    """
    Bounding box regression layer.
    """

    def __init__(self, in_channels, num_classes=2, bn_neck=True):
        """
        Args:
            in_channels (int): Input channels.
            num_classes (int, optional): Defaults to 2 (background and pedestrian).
            bn_neck (bool, optional): Whether to use BN after Linear. Defaults to True.
        """
        super(BBoxRegressor, self).__init__()
        if bn_neck:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes), nn.BatchNorm1d(4 * num_classes)
            )
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def detection_losses(
    box_cls_scores_1st,
    box_regs_1st,
    box_labels_1st,
    box_reg_targets_1st,
    box_cls_scores_2nd,
    box_regs_2nd=None,
    box_labels_2nd=None,
    box_reg_targets_2nd=None,
):
    # --------------------- The first stage -------------------- #
    box_labels_1st = torch.cat(box_labels_1st, dim=0)
    box_reg_targets_1st = torch.cat(box_reg_targets_1st, dim=0)
    loss_rcnn_cls_1st = F.cross_entropy(box_cls_scores_1st, box_labels_1st)

    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(box_labels_1st > 0).squeeze(1)
    labels_pos = box_labels_1st[sampled_pos_inds_subset]
    N = box_cls_scores_1st.size(0)
    box_regs_1st = box_regs_1st.reshape(N, -1, 4)

    loss_rcnn_reg_1st = F.smooth_l1_loss(
        box_regs_1st[sampled_pos_inds_subset, labels_pos],
        box_reg_targets_1st[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_rcnn_reg_1st = loss_rcnn_reg_1st / box_labels_1st.numel()

    # --------------------- The second stage -------------------- #
    if (
        box_regs_2nd is not None
        and box_labels_2nd is not None
        and box_reg_targets_2nd is not None
    ):
        box_labels_2nd = torch.cat(box_labels_2nd, dim=0)
        box_reg_targets_2nd = torch.cat(box_reg_targets_2nd, dim=0)
        loss_rcnn_cls_2nd = F.binary_cross_entropy_with_logits(
            box_cls_scores_2nd, box_labels_2nd.float()
        )

        sampled_pos_inds_subset = torch.nonzero(box_labels_2nd > 0).squeeze(1)
        labels_pos = box_labels_2nd[sampled_pos_inds_subset]
        N = box_cls_scores_2nd.size(0)
        box_regs_2nd = box_regs_2nd.reshape(N, -1, 4)

        loss_rcnn_reg_2nd = F.smooth_l1_loss(
            box_regs_2nd[sampled_pos_inds_subset, labels_pos],
            box_reg_targets_2nd[sampled_pos_inds_subset],
            reduction="sum",
        )
        loss_rcnn_reg_2nd = loss_rcnn_reg_2nd / box_labels_2nd.numel()
    else:
        loss_rcnn_cls_2nd = torch.tensor(
            0, dtype=loss_rcnn_cls_1st.dtype, device=loss_rcnn_cls_1st.device
        )
        loss_rcnn_reg_2nd = torch.tensor(
            0, dtype=loss_rcnn_reg_1st.dtype, device=loss_rcnn_reg_1st.device
        )

    return dict(
        loss_rcnn_cls_1st=loss_rcnn_cls_1st,
        loss_rcnn_reg_1st=loss_rcnn_reg_1st,
        loss_rcnn_cls_2nd=loss_rcnn_cls_2nd,
        loss_rcnn_reg_2nd=loss_rcnn_reg_2nd,
    )


@META_ARCH_REGISTRY.register()
class IF_COAT(BASE_COAT):
    def __init__(self, cfg):
        super(BASE_COAT, self).__init__()

        backbone, res5_head = coat_resnet.build_resnet(
            name="resnet50", pretrained=True
        )
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.DETECTOR.MODEL.RPN.PRE_NMS_TOPN_TRAIN,
            testing=cfg.DETECTOR.MODEL.RPN.PRE_NMS_TOPN_TEST,
        )
        post_nms_top_n = dict(
            training=cfg.DETECTOR.MODEL.RPN.POST_NMS_TOPN_TRAIN,
            testing=cfg.DETECTOR.MODEL.RPN.POST_NMS_TOPN_TEST,
        )
        rpn = RPNwithScores(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.DETECTOR.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.DETECTOR.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.DETECTOR.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.DETECTOR.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.DETECTOR.MODEL.RPN.NMS_THRESH,
        )
        if cfg.DETECTOR.MODEL.ROI_HEAD.USE_RES5:
            box_head = res5_head
        else:
            # trans head
            box_head = coat_trans.TransformerHeadMaxp(
                cfg=cfg.REID_HEAD,
                kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
            )
        reid_head = coat_trans.TransformerHeadAvgp(
            cfg=cfg.REID_HEAD,
            kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
        )

        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        box_predictor = BBoxRegressor(
            2048, num_classes=2, bn_neck=cfg.DETECTOR.MODEL.ROI_HEAD.BN_NECK
        )
        roi_heads = DenseTransROIHeads(
            cfg=cfg,
            # Cascade Transformer Head
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.DETECTOR.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.DETECTOR.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.DETECTOR.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )

        if self.training:
            transform = GeneralizedRCNNTransform(
                min_size=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )
        else:
            transform = GeneralizedRCNNTransform(
                min_size=cfg.INPUT.MIN_SIZE_TEST,
                max_size=cfg.INPUT.MAX_SIZE_TEST,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )  # only to use post-process

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        lw_cfg = cfg.DETECTOR.LOSS.LOSS_WEIGHTS
        self.lw_rpn_reg = lw_cfg.LW_RPN_REG
        self.lw_rpn_cls = lw_cfg.LW_RPN_CLS
        self.lw_rcnn_reg = lw_cfg.LW_RCNN_REG
        self.lw_rcnn_cls = lw_cfg.LW_RCNN_CLS
        self.lw_rcnn_reid = cfg.REID_HEAD.LOSS.LOSS_WEIGHTS.CONTRAST

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        self.vis_period = cfg.VIS_PERIOD

    def preproxess_input(self, input_list):
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
            if "img2_input" in input_dict:
                img2_dict = input_dict["img2_input"]
                img_paths.append(img2_dict["file_name"])
                img_names.append(img2_dict["image_id"])
                img_ts.append(img2_dict["image"].to(self.device))
                xyxy_box_t = torch.tensor(
                    img2_dict["boxes"],
                    dtype=torch.float32,
                    device=self.device,
                )  # xyxy
                img_boxes.append(xyxy_box_t)
                img_pids.append(img2_dict["ids"])
                img_aug_hws.append((img2_dict["height"], img2_dict["width"]))
                img_org_hws.append((img2_dict["org_height"], img2_dict["org_width"]))
                img_org_boxes.append(torch.tensor(img2_dict["org_boxes"]))

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


class DenseTransROIHeads(TransROIHeads):
    def __init__(self, cfg, faster_rcnn_predictor, reid_head, *args, **kwargs):
        super(TransROIHeads, self).__init__(*args, **kwargs)

        # ROI head
        self.use_diff_thresh = cfg.DETECTOR.MODEL.ROI_HEAD.USE_DIFF_THRESH
        self.nms_thresh_1st = cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST
        self.nms_thresh_2nd = cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST_2ND
        self.fg_iou_thresh_1st = cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN
        self.bg_iou_thresh_1st = cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN
        self.fg_iou_thresh_2nd = cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN_2ND
        self.bg_iou_thresh_2nd = cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_2ND

        # Regression head
        self.box_predictor_1st = faster_rcnn_predictor
        self.box_predictor_2nd = self.box_predictor

        # Transformer head
        self.box_head_1st = self.box_head
        self.box_head_2nd = reid_head

        # Feature embedding
        ehead_cfg = cfg.REID_HEAD.EMBEDDING_HEAD
        embedding_dim = ehead_cfg.EMBEDDING_DIM
        ehead_infeats = ehead_cfg.IN_FEAT
        feat_channels = {
            "before_trans": 1024,
            "after_trans": 2048,
            "feat_res4": 1024,
            "feat_res5": 2048,
        }
        self.embedding_head_2nd = NormAwareEmbedding(
            featmap_names=ehead_infeats,
            in_channels=[feat_channels[fn] for fn in ehead_infeats],
            dim=embedding_dim,
        )
        self.nae_det = cfg.REID_HEAD.EMBEDDING_HEAD.DET_SUP
        # contrast
        self.c_temp = cfg.REID_HEAD.LOSS.CONTRAST.TEMP
        self.proposal_src = cfg.REID_HEAD.PROPOSAL_SRC
        self.pos_nms = cfg.REID_HEAD.POS_NMS
        self.pos_nms_t = cfg.REID_HEAD.POS_NMS_THRED
        self.multi_pos_loss = cfg.REID_HEAD.LOSS.CONTRAST.MULTI_POS_LOSS

        # create the queue
        len_queue = cfg.REID_HEAD.LOSS.CONTRAST.CQ_SIZE
        self.register_buffer("queue", torch.randn(embedding_dim, len_queue))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        comm.synchronize()
        all_keys = comm.all_gather(keys)
        all_keys = [k.to(self.queue.device) for k in all_keys]
        keys = torch.cat(all_keys, dim=0)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        target_ptr = ptr + batch_size
        if target_ptr > self.queue.shape[1]:
            left_len = self.queue.shape[1] - ptr
            self.queue[:, ptr : ptr + left_len] = keys[:left_len].T
            self.queue[:, : batch_size - left_len] = keys[left_len:].T
        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr : ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.queue.shape[1]  # move pointer

        self.queue_ptr[0] = ptr

    def select_training_samples(self, proposals, proposal_scores, targets):

        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        gt_scores = [torch.ones_like(t["labels"], dtype=torch.float32) for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)
        prop_scores = [
            torch.cat((p_s, gt_s)) for p_s, gt_s in zip(proposal_scores, gt_scores)
        ]

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
            prop_scores[img_id] = prop_scores[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, prop_scores, matched_idxs, labels, regression_targets

    def forward(self, features, rpn_out, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            boxes (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        gt_det_2nd = None
        feats_reid_2nd = None
        targets_reid_2nd = None
        boxes, boxes_scores = rpn_out

        if self.training:
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_1st,
                    self.bg_iou_thresh_1st,
                    allow_low_quality_matches=False,
                )
            (
                boxes,
                box_scores,
                _,
                box_pid_labels_1st,
                box_reg_targets_1st,
            ) = self.select_training_samples(boxes, boxes_scores, targets)

        # ------------------- The first stage ------------------ # trans feat head + faster_rcnn_predictor
        box_features_1st = self.box_roi_pool(features, boxes, image_shapes)
        box_features_1st = self.box_head_1st(box_features_1st)
        if "after_trans" in box_features_1st:
            out_name = "after_trans"
        else:
            out_name = "feat_res5"
        box_cls_scores_1st, box_regs_1st = self.box_predictor_1st(
            box_features_1st[out_name]
        )
        del box_features_1st

        if self.training:
            if self.nae_det:
                boxes = self.get_boxes(box_regs_1st, boxes, image_shapes)
                boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
                if self.use_diff_thresh:
                    self.proposal_matcher = det_utils.Matcher(
                        self.fg_iou_thresh_2nd,
                        self.bg_iou_thresh_2nd,
                        allow_low_quality_matches=False,
                    )
                box_scores = F.softmax(box_cls_scores_1st.detach(), -1)[..., 1]
                box_scores = box_scores.split([b_img.shape[0] for b_img in boxes])
                (
                    boxes,
                    box_scores,
                    _,
                    box_pid_labels_2nd,
                    box_reg_targets_2nd,
                ) = self.select_training_samples(boxes, box_scores, targets)
                reid_boxes = boxes
                reid_box_scores = box_scores
                reid_box_ids = box_pid_labels_2nd
            else:
                n_imgs = len(boxes)
                reid_boxes, reid_box_scores, reid_box_ids = (
                    [[] for _ in range(n_imgs)],
                    [[] for _ in range(n_imgs)],
                    [[] for _ in range(n_imgs)],
                )
                # len_per_img = [0 for _ in range(n_imgs)]
                if self.proposal_src == "merge" or self.proposal_src == "rpn":
                    # from rpn
                    for ni in range(n_imgs):
                        valid_mask = box_pid_labels_1st[ni] > 0
                        valid_boxes = boxes[ni][valid_mask].detach()
                        valid_box_scores = box_scores[ni][valid_mask].detach()
                        valid_labels = box_pid_labels_1st[ni][valid_mask]
                        reid_boxes[ni].append(valid_boxes)
                        reid_box_scores[ni].append(valid_box_scores)
                        reid_box_ids[ni].append(valid_labels)
                        # len_per_img[ni] += valid_mask.sum()

                boxes = self.get_boxes(box_regs_1st, boxes, image_shapes)
                boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
                if self.use_diff_thresh:
                    self.proposal_matcher = det_utils.Matcher(
                        self.fg_iou_thresh_2nd,
                        self.bg_iou_thresh_2nd,
                        allow_low_quality_matches=False,
                    )
                box_scores = F.softmax(box_cls_scores_1st.detach(), -1)[..., 1]
                box_scores = box_scores.split([b_img.shape[0] for b_img in boxes])
                (
                    boxes,
                    box_scores,
                    _,
                    box_pid_labels_2nd,
                    box_reg_targets_2nd,
                ) = self.select_training_samples(boxes, box_scores, targets)
                if self.proposal_src == "merge" or self.proposal_src == "det":
                    # from det
                    for ni in range(n_imgs):
                        valid_mask = box_pid_labels_2nd[ni] > 0
                        valid_boxes = boxes[ni][valid_mask].detach()
                        valid_box_scores = box_scores[ni][valid_mask].detach()
                        valid_labels = box_pid_labels_2nd[ni][valid_mask]
                        reid_boxes[ni].append(valid_boxes)
                        reid_box_scores[ni].append(valid_box_scores)
                        reid_box_ids[ni].append(valid_labels)
                        # len_per_img[ni] += valid_mask.sum()
                reid_boxes = [torch.cat(boxes_img, dim=0) for boxes_img in reid_boxes]
                reid_box_ids = [torch.cat(ids_img, dim=0) for ids_img in reid_box_ids]
                reid_box_scores = [
                    torch.cat(scores_img, dim=0) for scores_img in reid_box_scores
                ]
                if self.pos_nms:
                    for i, (r_boxes, r_ids, r_scores) in enumerate(
                        zip(reid_boxes, reid_box_ids, reid_box_scores)
                    ):
                        keep = box_ops.batched_nms(
                            r_boxes, r_scores, r_ids, self.pos_nms_t
                        )
                        reid_boxes[i] = r_boxes[keep]
                        reid_box_ids[i] = r_ids[keep]
                        reid_box_scores[i] = r_scores[keep]

        else:
            orig_thresh = self.nms_thresh  # 0.4
            self.nms_thresh = self.nms_thresh_1st
            reid_boxes, scores, _ = self.postprocess_proposals(
                box_cls_scores_1st, box_regs_1st, boxes, image_shapes
            )

        # no detection predicted by Faster R-CNN head in test phase
        if reid_boxes[0].shape[0] == 0:
            assert not self.training
            boxes_i = torch.zeros(0, 4)
            labels_i = torch.zeros(0)
            scores_i = torch.zeros(0)
            embeddings_i = torch.zeros(0, 256)
            return [
                dict(boxes=boxes_i, scores=scores_i, embeddings=embeddings_i)
                for _ in range(len(boxes))
            ], []

        # --------------------- The second stage -------------------- #
        box_features = self.box_roi_pool(features, reid_boxes, image_shapes)
        box_features = self.box_head_2nd(box_features)
        if "after_trans" in box_features:
            out_name = "after_trans"
        else:
            out_name = "feat_res5"
        box_regs_2nd = self.box_predictor_2nd(box_features[out_name])

        box_embeddings_2nd, box_cls_scores_2nd = self.embedding_head_2nd(
            {out_name: box_features[out_name]}
        )  # embeddings already l2 normed
        del box_features
        if box_cls_scores_2nd.dim() == 0:
            box_cls_scores_2nd = box_cls_scores_2nd.unsqueeze(0)

        result, losses = [], {}
        if self.training:
            box_labels_1st = [y.clamp(0, 1) for y in box_pid_labels_1st]
            box_labels_2nd = [y.clamp(0, 1) for y in box_pid_labels_2nd]
            if self.nae_det:
                losses = detection_losses(
                    box_cls_scores_1st,
                    box_regs_1st,
                    box_labels_1st,
                    box_reg_targets_1st,
                    box_cls_scores_2nd,
                    box_regs_2nd,
                    box_labels_2nd,
                    box_reg_targets_2nd,
                )
            else:
                losses = detection_losses(
                    box_cls_scores_1st,
                    box_regs_1st,
                    box_labels_1st,
                    box_reg_targets_1st,
                    box_cls_scores_2nd,
                    None,
                    None,
                    None,
                )
            if self.nae_det:
                pos_ids = []
                pos_embds = []
                pos_boxes = []
                boxes = self.get_boxes(box_regs_2nd, reid_boxes, image_shapes)
                boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
                box_fcs_scores = box_scores
                box_embs = box_embeddings_2nd.split([b_img.shape[0] for b_img in boxes])
                for i, (i_boxes, i_ids, i_scores, i_embs) in enumerate(
                    zip(boxes, box_pid_labels_2nd, box_fcs_scores, box_embs)
                ):
                    p_mask = i_ids > 0
                    p_ids = i_ids[p_mask]
                    p_embs = i_embs[p_mask]
                    p_boxes = i_boxes[p_mask]
                    if self.pos_nms:
                        p_scores = i_scores[p_mask]
                        keep = box_ops.batched_nms(
                            p_boxes, p_scores, p_ids, self.pos_nms_t
                        )
                        p_ids = p_ids[keep]
                        p_embs = p_embs[keep]
                        p_boxes = p_boxes[keep]
                    pos_ids.append(p_ids)
                    pos_embds.append(p_embs)
                    pos_boxes.append(p_boxes)
                ct_ids = torch.cat(pos_ids, dim=0)
                ct_embds = torch.cat(pos_embds, dim=0)
            else:
                ct_ids = torch.cat(reid_box_ids, dim=0)
                ct_embds = box_embeddings_2nd
                pos_ids = reid_box_ids
                pos_boxes = reid_boxes
            l_neg_all = (
                torch.einsum("nc,ck->nk", [ct_embds, self.queue.clone().detach()])
                / self.c_temp
            )
            # iter by id
            uniqe_pids = torch.unique(ct_ids)
            p_losses = []
            for pid in uniqe_pids:
                pid_mask = ct_ids == pid
                # NOTE ignore single for now
                if pid_mask.sum() < 2:
                    continue
                pid_embeddings = ct_embds[pid_mask]
                l_pos = (
                    torch.einsum("nc,ck->nk", [pid_embeddings, pid_embeddings.T])
                    / self.c_temp
                )  # self similarity is included, np x np
                l_neg = l_neg_all[pid_mask]  # np x ng
                np, ng = l_pos.shape[0], l_neg.shape[1]
                select_cvt = torch.eye(np, dtype=torch.bool, device=self.queue.device)
                select_mask = torch.logical_not(select_cvt)
                l_neg_expd = l_neg.unsqueeze(1).expand(-1, np - 1, -1)
                if self.multi_pos_loss:
                    l_pos_r = torch.stack(l_pos[select_mask].split(np - 1)).unsqueeze(2)
                    l_pos_expd = l_pos_r.expand(-1, -1, ng)  # np x np-1 x ng
                    diff = torch.exp(l_neg_expd - l_pos_expd)  # np x np-1 x ng
                    exp_sum = diff.sum(-1).sum(-1)  # np
                    id_losses = torch.log(1 + exp_sum)
                else:
                    l_pos_expd = l_pos[select_mask].unsqueeze(1)  # (np x np-1) x 1
                    l_neg_expd = l_neg_expd.reshape(-1, ng)
                    l_all = torch.cat([l_pos_expd, l_neg_expd], dim=-1)
                    labels = torch.zeros(
                        l_all.shape[0], dtype=torch.long, device=self.queue.device
                    )  # (np x np-1)
                    id_losses = F.cross_entropy(
                        l_all, labels, reduction="none"
                    )  # (np x np-1)
                    id_losses = id_losses.view(np, np - 1)
                    id_losses = id_losses.mean(dim=1)

                p_losses.append(id_losses)
            p_losses = torch.cat(p_losses, dim=0)
            # try to reduce mem usage
            del l_neg_all
            del l_pos_expd
            del l_pos
            del l_neg_expd
            del l_neg
            del id_losses
            del l_all
            n_inst = torch.as_tensor(
                [p_losses.shape[0]], dtype=torch.float, device=self.queue.device
            )
            comm.synchronize()
            all_n_inst = comm.all_gather(n_inst)
            num_inst = sum([num.to("cpu") for num in all_n_inst])
            num_inst = torch.clamp(num_inst / comm.get_world_size(), min=1).item()
            reid_loss = p_losses.sum() / num_inst

            losses.update(loss_rcnn_reid_2nd=reid_loss)
            del p_losses
            self._dequeue_and_enqueue(box_embeddings_2nd)
            # for vis only
            for i in range(len(pos_boxes)):
                result.append(
                    dict(
                        boxes=pos_boxes[i],
                        labels=pos_ids[i],
                    )
                )

        else:
            if self.nae_det:
                boxes, scores, embeddings_2nd, labels = self.postprocess_boxes(
                    box_cls_scores_2nd,
                    box_regs_2nd,
                    box_embeddings_2nd,
                    reid_boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=None,
                    cws=False,
                )
            else:
                boxes = reid_boxes
                embeddings_2nd = torch.split(
                    box_embeddings_2nd, [bx.shape[0] for bx in boxes]
                )
            num_images = len(boxes)
            for i in range(num_images):
                embeddings = embeddings_2nd[i]
                result.append(
                    dict(
                        boxes=boxes[i],
                        scores=scores[i],
                        embeddings=embeddings,
                    )
                )

        return (
            result,
            losses,
            feats_reid_2nd,
            targets_reid_2nd,
        )


@META_ARCH_REGISTRY.register()
class IF_COAT_SIM(IF_COAT):
    def __init__(self, cfg):
        super(BASE_COAT, self).__init__()

        backbone, res5_head = coat_resnet.build_resnet(name="resnet50", pretrained=True)
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.DETECTOR.MODEL.RPN.PRE_NMS_TOPN_TRAIN,
            testing=cfg.DETECTOR.MODEL.RPN.PRE_NMS_TOPN_TEST,
        )
        post_nms_top_n = dict(
            training=cfg.DETECTOR.MODEL.RPN.POST_NMS_TOPN_TRAIN,
            testing=cfg.DETECTOR.MODEL.RPN.POST_NMS_TOPN_TEST,
        )
        rpn = RPNwithScores(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.DETECTOR.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.DETECTOR.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.DETECTOR.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.DETECTOR.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.DETECTOR.MODEL.RPN.NMS_THRESH,
        )
        if cfg.DETECTOR.MODEL.ROI_HEAD.USE_RES5:
            box_head = res5_head
        else:
            # trans head
            box_head = TransformerHeadSimBox(
                cfg=cfg.REID_HEAD,
                kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
            )
        reid_head = TransformerHeadSimReid(
            cfg=cfg.REID_HEAD,
            kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
        )

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        box_predictor = (
            FastRCNNPredictor(2048, 2)
            if cfg.DETECTOR.MODEL.ROI_HEAD.USE_RES5
            else FastRCNNPredictor(cfg.REID_HEAD.DIM_MODEL, 2)
        )
        roi_heads = DenseTransROIHeadsSim(
            cfg=cfg,
            # Cascade Transformer Head
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.DETECTOR.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.DETECTOR.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.DETECTOR.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )

        if self.training:
            transform = GeneralizedRCNNTransform(
                min_size=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )
        else:
            transform = GeneralizedRCNNTransform(
                min_size=cfg.INPUT.MIN_SIZE_TEST,
                max_size=cfg.INPUT.MAX_SIZE_TEST,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )  # only to use post-process

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        lw_cfg = cfg.DETECTOR.LOSS.LOSS_WEIGHTS
        self.lw_rpn_reg = lw_cfg.LW_RPN_REG
        self.lw_rpn_cls = lw_cfg.LW_RPN_CLS
        self.lw_rcnn_reg = lw_cfg.LW_RCNN_REG
        self.lw_rcnn_cls = lw_cfg.LW_RCNN_CLS
        self.lw_rcnn_reid = cfg.REID_HEAD.LOSS.LOSS_WEIGHTS.CONTRAST

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        self.vis_period = cfg.VIS_PERIOD

    def inference(self, input_list):
        if "query" in input_list[0]:
            input_batches = self.preproxess_input([qd["query"] for qd in input_list])
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
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(
                features, boxes, images.image_sizes
            )
            box_features_2nd = self.roi_heads.box_head_2nd(box_features)
            if "after_trans" in box_features_2nd:
                out_name = "after_trans"
            else:
                out_name = "feat_res5"
            box_embeddings_2nd = (
                box_features_2nd[out_name].squeeze(-1).squeeze(-1)
            )  # not l2 normed
            box_embeddings_2nd = self.roi_heads.bn_neck(box_embeddings_2nd)
            box_embeddings_2nd = F.normalize(box_embeddings_2nd, dim=1)
            embeddings = box_embeddings_2nd.cpu()  # .split(1, 0)
            result_list = input_list.copy()
            for bi, feat in enumerate(embeddings):
                result_list[bi]["query"]["feat"] = feat.view(-1)
            return result_list
        else:
            # gallery
            return super().inference(input_list)


@META_ARCH_REGISTRY.register()
class BASE_COAT_SIM(IF_COAT_SIM):
    def __init__(self, cfg):
        super(BASE_COAT, self).__init__()

        backbone, res5_head = coat_resnet.build_resnet(name="resnet50", pretrained=True)
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.DETECTOR.MODEL.RPN.PRE_NMS_TOPN_TRAIN,
            testing=cfg.DETECTOR.MODEL.RPN.PRE_NMS_TOPN_TEST,
        )
        post_nms_top_n = dict(
            training=cfg.DETECTOR.MODEL.RPN.POST_NMS_TOPN_TRAIN,
            testing=cfg.DETECTOR.MODEL.RPN.POST_NMS_TOPN_TEST,
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.DETECTOR.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.DETECTOR.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.DETECTOR.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.DETECTOR.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.DETECTOR.MODEL.RPN.NMS_THRESH,
        )
        if cfg.DETECTOR.MODEL.ROI_HEAD.USE_RES5:
            box_head = res5_head
        else:
            # trans head
            box_head = TransformerHeadSimBox(
                cfg=cfg.REID_HEAD,
                kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
            )
        reid_head = TransformerHeadSimReid(
            cfg=cfg.REID_HEAD,
            kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
        )

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        box_predictor = (
            FastRCNNPredictor(2048, 2)
            if cfg.DETECTOR.MODEL.ROI_HEAD.USE_RES5
            else FastRCNNPredictor(cfg.REID_HEAD.DIM_MODEL, 2)
        )
        roi_heads = TransROIHeadsSim(
            cfg=cfg,
            # Cascade Transformer Head
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.DETECTOR.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.DETECTOR.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.DETECTOR.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )

        if self.training:
            transform = GeneralizedRCNNTransform(
                min_size=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )
        else:
            transform = GeneralizedRCNNTransform(
                min_size=cfg.INPUT.MIN_SIZE_TEST,
                max_size=cfg.INPUT.MAX_SIZE_TEST,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )  # only to use post-process

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        lw_cfg = cfg.DETECTOR.LOSS.LOSS_WEIGHTS
        self.lw_rpn_reg = lw_cfg.LW_RPN_REG
        self.lw_rpn_cls = lw_cfg.LW_RPN_CLS
        self.lw_rcnn_reg = lw_cfg.LW_RCNN_REG
        self.lw_rcnn_cls = lw_cfg.LW_RCNN_CLS
        self.lw_rcnn_reid = cfg.REID_HEAD.LOSS.LOSS_WEIGHTS.OIM

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        self.vis_period = cfg.VIS_PERIOD


class DenseTransROIHeadsSim(DenseTransROIHeads):
    def __init__(self, cfg, reid_head, *args, **kwargs):
        super(TransROIHeads, self).__init__(*args, **kwargs)

        # ROI head
        self.use_diff_thresh = cfg.DETECTOR.MODEL.ROI_HEAD.USE_DIFF_THRESH
        self.nms_thresh_1st = cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST
        self.nms_thresh_2nd = cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST_2ND
        self.fg_iou_thresh_1st = cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN
        self.bg_iou_thresh_1st = cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN
        self.fg_iou_thresh_2nd = cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN_2ND
        self.bg_iou_thresh_2nd = cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_2ND

        # Regression head
        self.box_predictor_1st = self.box_predictor

        # Transformer head
        self.box_head_1st = self.box_head
        self.box_head_2nd = reid_head

        # contrast
        self.c_temp = cfg.REID_HEAD.LOSS.CONTRAST.TEMP
        self.proposal_src = cfg.REID_HEAD.PROPOSAL_SRC
        self.pos_nms = cfg.REID_HEAD.POS_NMS
        self.pos_nms_t = cfg.REID_HEAD.POS_NMS_THRED
        self.multi_pos_loss = cfg.REID_HEAD.LOSS.CONTRAST.MULTI_POS_LOSS

        # create the queue
        len_queue = cfg.REID_HEAD.LOSS.CONTRAST.CQ_SIZE
        self.register_buffer("queue", torch.randn(cfg.REID_HEAD.DIM_MODEL, len_queue))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections
        self.bn_neck = nn.BatchNorm1d(cfg.REID_HEAD.DIM_MODEL)

    def forward(self, features, rpn_out, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            boxes (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        gt_det_2nd = None
        feats_reid_2nd = None
        targets_reid_2nd = None
        boxes, boxes_scores = rpn_out

        if self.training:
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_1st,
                    self.bg_iou_thresh_1st,
                    allow_low_quality_matches=False,
                )
            (
                boxes,
                box_scores,
                _,
                box_pid_labels_1st,
                box_reg_targets_1st,
            ) = self.select_training_samples(boxes, boxes_scores, targets)

        # ------------------- The first stage ------------------ # trans feat head + faster_rcnn_predictor
        box_features_1st = self.box_roi_pool(features, boxes, image_shapes)
        box_features_1st = self.box_head_1st(box_features_1st)
        if "after_trans" in box_features_1st:
            out_name = "after_trans"
        else:
            out_name = "feat_res5"
        box_cls_scores_1st, box_regs_1st = self.box_predictor_1st(
            box_features_1st[out_name]
        )
        del box_features_1st

        if self.training:
            n_imgs = len(boxes)
            reid_boxes, reid_box_scores, reid_box_ids = (
                [[] for _ in range(n_imgs)],
                [[] for _ in range(n_imgs)],
                [[] for _ in range(n_imgs)],
            )
            # len_per_img = [0 for _ in range(n_imgs)]
            if self.proposal_src == "merge" or self.proposal_src == "rpn":
                # from rpn
                for ni in range(n_imgs):
                    valid_mask = box_pid_labels_1st[ni] > 0
                    valid_boxes = boxes[ni][valid_mask].detach()
                    valid_box_scores = box_scores[ni][valid_mask].detach()
                    valid_labels = box_pid_labels_1st[ni][valid_mask]
                    reid_boxes[ni].append(valid_boxes)
                    reid_box_scores[ni].append(valid_box_scores)
                    reid_box_ids[ni].append(valid_labels)
                    # len_per_img[ni] += valid_mask.sum()

            boxes = self.get_boxes(box_regs_1st, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_2nd,
                    self.bg_iou_thresh_2nd,
                    allow_low_quality_matches=False,
                )
            box_scores = F.softmax(box_cls_scores_1st.detach(), -1)[..., 1]
            box_scores = box_scores.split([b_img.shape[0] for b_img in boxes])
            (
                boxes,
                box_scores,
                _,
                box_pid_labels_2nd,
                box_reg_targets_2nd,
            ) = self.select_training_samples(boxes, box_scores, targets)
            if self.proposal_src == "merge" or self.proposal_src == "det":
                # from det
                for ni in range(n_imgs):
                    valid_mask = box_pid_labels_2nd[ni] > 0
                    valid_boxes = boxes[ni][valid_mask].detach()
                    valid_box_scores = box_scores[ni][valid_mask].detach()
                    valid_labels = box_pid_labels_2nd[ni][valid_mask]
                    reid_boxes[ni].append(valid_boxes)
                    reid_box_scores[ni].append(valid_box_scores)
                    reid_box_ids[ni].append(valid_labels)
                    # len_per_img[ni] += valid_mask.sum()
            reid_boxes = [torch.cat(boxes_img, dim=0) for boxes_img in reid_boxes]
            reid_box_ids = [torch.cat(ids_img, dim=0) for ids_img in reid_box_ids]
            reid_box_scores = [
                torch.cat(scores_img, dim=0) for scores_img in reid_box_scores
            ]
            if self.pos_nms:
                for i, (r_boxes, r_ids, r_scores) in enumerate(
                    zip(reid_boxes, reid_box_ids, reid_box_scores)
                ):
                    keep = box_ops.batched_nms(r_boxes, r_scores, r_ids, self.pos_nms_t)
                    reid_boxes[i] = r_boxes[keep]
                    reid_box_ids[i] = r_ids[keep]
                    reid_box_scores[i] = r_scores[keep]

        else:
            orig_thresh = self.nms_thresh  # 0.4
            self.nms_thresh = self.nms_thresh_1st
            reid_boxes, scores, _ = self.postprocess_proposals(
                box_cls_scores_1st, box_regs_1st, boxes, image_shapes
            )

        # no detection predicted by Faster R-CNN head in test phase
        if reid_boxes[0].shape[0] == 0:
            assert not self.training
            boxes_i = torch.zeros(0, 4)
            labels_i = torch.zeros(0)
            scores_i = torch.zeros(0)
            embeddings_i = torch.zeros(0, 256)
            return [
                dict(boxes=boxes_i, scores=scores_i, embeddings=embeddings_i)
                for _ in range(len(boxes))
            ], []

        # --------------------- The second stage -------------------- #
        box_features = self.box_roi_pool(features, reid_boxes, image_shapes)
        box_features = self.box_head_2nd(box_features)
        if "after_trans" in box_features:
            out_name = "after_trans"
        else:
            out_name = "feat_res5"
        box_embeddings_2nd = (
            box_features[out_name].squeeze(-1).squeeze(-1)
        )  # not l2 normed
        box_embeddings_2nd = self.bn_neck(box_embeddings_2nd)
        box_embeddings_2nd = F.normalize(box_embeddings_2nd, dim=1)
        # embeddings already l2 normed
        del box_features

        result, losses = [], {}
        if self.training:
            box_labels_1st = [y.clamp(0, 1) for y in box_pid_labels_1st]
            losses = detection_losses(
                box_cls_scores_1st,
                box_regs_1st,
                box_labels_1st,
                box_reg_targets_1st,
                None,
                None,
                None,
                None,
            )
            ct_ids = torch.cat(reid_box_ids, dim=0)
            ct_embds = box_embeddings_2nd
            pos_ids = reid_box_ids
            pos_boxes = reid_boxes
            l_neg_all = (
                torch.einsum("nc,ck->nk", [ct_embds, self.queue.clone().detach()])
                / self.c_temp
            )
            # iter by id
            uniqe_pids = torch.unique(ct_ids)
            p_losses = []
            for pid in uniqe_pids:
                pid_mask = ct_ids == pid
                # NOTE ignore single for now
                if pid_mask.sum() < 2:
                    continue
                pid_embeddings = ct_embds[pid_mask]
                l_pos = (
                    torch.einsum("nc,ck->nk", [pid_embeddings, pid_embeddings.T])
                    / self.c_temp
                )  # self similarity is included, np x np
                l_neg = l_neg_all[pid_mask]  # np x ng
                np, ng = l_pos.shape[0], l_neg.shape[1]
                select_cvt = torch.eye(np, dtype=torch.bool, device=self.queue.device)
                select_mask = torch.logical_not(select_cvt)
                l_neg_expd = l_neg.unsqueeze(1).expand(-1, np - 1, -1)
                if self.multi_pos_loss:
                    l_pos_r = torch.stack(l_pos[select_mask].split(np - 1)).unsqueeze(2)
                    l_pos_expd = l_pos_r.expand(-1, -1, ng)  # np x np-1 x ng
                    diff = torch.exp(l_neg_expd - l_pos_expd)  # np x np-1 x ng
                    exp_sum = diff.sum(-1).sum(-1)  # np
                    id_losses = torch.log(1 + exp_sum)
                else:
                    l_pos_expd = l_pos[select_mask].unsqueeze(1)  # (np x np-1) x 1
                    l_neg_expd = l_neg_expd.reshape(-1, ng)
                    l_all = torch.cat([l_pos_expd, l_neg_expd], dim=-1)
                    labels = torch.zeros(
                        l_all.shape[0], dtype=torch.long, device=self.queue.device
                    )  # (np x np-1)
                    id_losses = F.cross_entropy(
                        l_all, labels, reduction="none"
                    )  # (np x np-1)
                    id_losses = id_losses.view(np, np - 1)
                    id_losses = id_losses.mean(dim=1)

                p_losses.append(id_losses)
            p_losses = torch.cat(p_losses, dim=0)
            # try to reduce mem usage
            del l_neg_all
            del l_pos_expd
            del l_pos
            del l_neg_expd
            del l_neg
            del id_losses
            del l_all
            n_inst = torch.as_tensor(
                [p_losses.shape[0]], dtype=torch.float, device=self.queue.device
            )
            comm.synchronize()
            all_n_inst = comm.all_gather(n_inst)
            num_inst = sum([num.to("cpu") for num in all_n_inst])
            num_inst = torch.clamp(num_inst / comm.get_world_size(), min=1).item()
            reid_loss = p_losses.sum() / num_inst

            losses.update(loss_rcnn_reid_2nd=reid_loss)
            del p_losses
            self._dequeue_and_enqueue(box_embeddings_2nd)
            # for vis only
            for i in range(len(pos_boxes)):
                result.append(
                    dict(
                        boxes=pos_boxes[i],
                        labels=pos_ids[i],
                    )
                )

        else:
            boxes = reid_boxes
            embeddings_2nd = torch.split(
                box_embeddings_2nd, [bx.shape[0] for bx in boxes]
            )
            num_images = len(boxes)
            for i in range(num_images):
                embeddings = embeddings_2nd[i]
                result.append(
                    dict(
                        boxes=boxes[i],
                        scores=scores[i],
                        embeddings=embeddings,
                    )
                )

        return (
            result,
            losses,
            feats_reid_2nd,
            targets_reid_2nd,
        )


from torchvision.models.detection.rpn import concat_box_prediction_layers


class RPNwithScores(RegionProposalNetwork):
    def forward(
        self,
        images,
        features,
        targets=None,
    ):

        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (OrderedDict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [
            s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors
        ]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(
            objectness, pred_bbox_deltas
        )
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(
            proposals, objectness, images.image_sizes, num_anchors_per_level
        )

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return (boxes, scores), losses


@META_ARCH_REGISTRY.register()
class COAT_L2P(BASE_COAT):
    def __init__(self, cfg):
        super(BASE_COAT, self).__init__()

        backbone, res5_head = coat_resnet.build_resnet(
            name="resnet50", pretrained=False
        )
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.DETECTOR.MODEL.RPN.PRE_NMS_TOPN_TRAIN,
            testing=cfg.DETECTOR.MODEL.RPN.PRE_NMS_TOPN_TEST,
        )
        post_nms_top_n = dict(
            training=cfg.DETECTOR.MODEL.RPN.POST_NMS_TOPN_TRAIN,
            testing=cfg.DETECTOR.MODEL.RPN.POST_NMS_TOPN_TEST,
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.DETECTOR.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.DETECTOR.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.DETECTOR.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.DETECTOR.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.DETECTOR.MODEL.RPN.NMS_THRESH,
        )
        if cfg.DETECTOR.MODEL.ROI_HEAD.USE_RES5:
            box_head = res5_head
        else:
            # trans head
            if cfg.REID_HEAD.L2P.DET_L2P:
                box_head = TransformerHeadL2P(
                    cfg=cfg.REID_HEAD,
                    kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
                )
            else:
                box_head = coat_trans.TransformerHead(
                    cfg=cfg.REID_HEAD,
                    kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
                )
        reid_head = TransformerHeadL2P(
            cfg=cfg.REID_HEAD,
            kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
        )

        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        box_predictor = BBoxRegressor(
            2048, num_classes=2, bn_neck=cfg.DETECTOR.MODEL.ROI_HEAD.BN_NECK
        )
        roi_heads = TransROIHeadsL2P(
            cfg=cfg,
            # Cascade Transformer Head
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.DETECTOR.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.DETECTOR.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.DETECTOR.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )

        if self.training:
            transform = GeneralizedRCNNTransform(
                min_size=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )
        else:
            transform = GeneralizedRCNNTransform(
                min_size=cfg.INPUT.MIN_SIZE_TEST,
                max_size=cfg.INPUT.MAX_SIZE_TEST,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )  # only to use post-process

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        lw_cfg = cfg.DETECTOR.LOSS.LOSS_WEIGHTS
        self.lw_rpn_reg = lw_cfg.LW_RPN_REG
        self.lw_rpn_cls = lw_cfg.LW_RPN_CLS
        self.lw_rcnn_reg = lw_cfg.LW_RCNN_REG
        self.lw_rcnn_cls = lw_cfg.LW_RCNN_CLS
        self.lw_rcnn_reid = cfg.REID_HEAD.LOSS.LOSS_WEIGHTS.OIM
        self.lw_l2p = cfg.REID_HEAD.LOSS.LOSS_WEIGHTS.L2P

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        self.vis_period = cfg.VIS_PERIOD

    def forward(self, input_list):
        res = super().forward(input_list)
        if self.training:
            for lk in res.keys():
                if "loss_l2p" in lk:
                    res[lk] *= self.lw_l2p
        return res

    def freeze_params(self):
        exclude_name = ["roi_heads"]

        def is_exluced(pn):
            exc = False
            for en in exclude_name:
                if en in pn:
                    exc = True
                    break
            return exc

        for name, param in self.named_parameters():
            if not is_exluced(name):
                param.requires_grad = False
        self.roi_heads.freeze_params()


class TransROIHeadsL2P(TransROIHeads):
    def __init__(self, cfg, faster_rcnn_predictor, reid_head, *args, **kwargs):
        super().__init__(cfg, faster_rcnn_predictor, reid_head, *args, **kwargs)
        if isinstance(self.box_head_1st, TransformerHeadL2P):
            self.det_l2p = True
        else:
            self.det_l2p = False

    def freeze_params(self):
        exclude_name = ["box_head_2nd"]

        def is_exluced(pn):
            exc = False
            for en in exclude_name:
                if en in pn:
                    exc = True
                    break
            return exc

        for name, param in self.named_parameters():
            if not is_exluced(name):
                param.requires_grad = False

    def forward(self, features, boxes, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            boxes (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        gt_det_2nd = None
        feats_reid_2nd = None
        targets_reid_2nd = None

        if self.training:
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_1st,
                    self.bg_iou_thresh_1st,
                    allow_low_quality_matches=False,
                )
            (
                boxes,
                _,
                box_pid_labels_1st,
                box_reg_targets_1st,
            ) = self.select_training_samples(
                boxes, targets
            )  # 2000 input from rpn

        # ------------------- The first stage ------------------ # trans feat head + faster_rcnn_predictor
        box_features_1st = self.box_roi_pool(features, boxes, image_shapes)
        if self.det_l2p:
            box_features_1st, det_l2p_l = self.box_head_1st(box_features_1st)
        else:
            box_features_1st = self.box_head_1st(box_features_1st)
            det_l2p_l = None
        if "after_trans" in box_features_1st:
            out_name = "after_trans"
        else:
            out_name = "feat_res5"
        box_cls_scores_1st, box_regs_1st = self.box_predictor_1st(
            box_features_1st[out_name]
        )
        del box_features_1st

        if self.training:
            boxes = self.get_boxes(box_regs_1st, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_2nd,
                    self.bg_iou_thresh_2nd,
                    allow_low_quality_matches=False,
                )
            (
                boxes,
                _,
                box_pid_labels_2nd,
                box_reg_targets_2nd,
            ) = self.select_training_samples(boxes, targets)
        else:
            orig_thresh = self.nms_thresh  # 0.4
            self.nms_thresh = self.nms_thresh_1st
            boxes, scores, _ = self.postprocess_proposals(
                box_cls_scores_1st, box_regs_1st, boxes, image_shapes
            )
            self.nms_thresh = orig_thresh

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes_i = torch.zeros(0, 4)
            labels_i = torch.zeros(0)
            scores_i = torch.zeros(0)
            embeddings_i = torch.zeros(0, 256)
            return [
                dict(
                    boxes=boxes_i,
                    labels=labels_i,
                    scores=scores_i,
                    embeddings=embeddings_i,
                )
                for _ in range(len(boxes))
            ], []

        # --------------------- The second stage -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features, l2p_l = self.box_head_2nd(box_features)
        if "after_trans" in box_features:
            out_name = "after_trans"
        else:
            out_name = "feat_res5"
        box_regs_2nd = self.box_predictor_2nd(box_features[out_name])
        box_embeddings_2nd, box_cls_scores_2nd = self.embedding_head_2nd(
            {out_name: box_features[out_name]}
        )
        if box_cls_scores_2nd.dim() == 0:
            box_cls_scores_2nd = box_cls_scores_2nd.unsqueeze(0)
        del box_features
        result, losses = [], {}
        if self.training:
            box_labels_1st = [y.clamp(0, 1) for y in box_pid_labels_1st]
            box_labels_2nd = [y.clamp(0, 1) for y in box_pid_labels_2nd]
            if self.nae_det:
                losses = detection_losses(
                    box_cls_scores_1st,
                    box_regs_1st,
                    box_labels_1st,
                    box_reg_targets_1st,
                    box_cls_scores_2nd,
                    box_regs_2nd,
                    box_labels_2nd,
                    box_reg_targets_2nd,
                )
            else:
                losses = detection_losses(
                    box_cls_scores_1st,
                    box_regs_1st,
                    box_labels_1st,
                    box_reg_targets_1st,
                    box_cls_scores_2nd,
                    None,
                    None,
                    None,
                )

            (
                loss_rcnn_reid_2nd,
                feats_reid_2nd,
                targets_reid_2nd,
                valid_inds,
            ) = self.reid_loss_2nd(box_embeddings_2nd, box_pid_labels_2nd)
            losses.update(loss_rcnn_reid_2nd=loss_rcnn_reid_2nd)
            losses.update(l2p_l)
            if det_l2p_l is not None:
                losses.update(det_l2p_l)
            # for vis only
            valid_inds_per_img = torch.split(
                valid_inds, [y.shape[0] for y in box_pid_labels_2nd]
            )
            valid_boxes = [b[vi] for b, vi in zip(boxes, valid_inds_per_img)]
            pid_labels_per_img = [
                pid[vi] for pid, vi in zip(box_pid_labels_2nd, valid_inds_per_img)
            ]
            for i in range(len(boxes)):
                result.append(
                    dict(
                        boxes=valid_boxes[i],
                        labels=pid_labels_per_img[i],
                    )
                )

        else:
            if self.nae_det:
                boxes, scores, embeddings_2nd, labels = self.postprocess_boxes(
                    box_cls_scores_2nd,
                    box_regs_2nd,
                    box_embeddings_2nd,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=None,
                    cws=False,
                )
            else:

                embeddings_2nd = torch.split(
                    box_embeddings_2nd, [bx.shape[0] for bx in boxes]
                )
            num_images = len(boxes)
            for i in range(num_images):
                embeddings = embeddings_2nd[i]
                result.append(
                    dict(
                        boxes=boxes[i],
                        scores=scores[i],
                        embeddings=embeddings,
                    )
                )
                num_images = len(boxes)
                for i in range(num_images):
                    embeddings = embeddings_2nd[i]
                    result.append(
                        dict(
                            boxes=boxes[i],
                            scores=scores[i],
                            embeddings=embeddings,
                        )
                    )

        return (
            result,
            losses,
            feats_reid_2nd,
            targets_reid_2nd,
        )


@META_ARCH_REGISTRY.register()
class COAT_SIM_L2P(BASE_COAT):
    def __init__(self, cfg):
        super(BASE_COAT, self).__init__()

        backbone, res5_head = coat_resnet.build_resnet(
            name="resnet50", pretrained=False
        )
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.DETECTOR.MODEL.RPN.PRE_NMS_TOPN_TRAIN,
            testing=cfg.DETECTOR.MODEL.RPN.PRE_NMS_TOPN_TEST,
        )
        post_nms_top_n = dict(
            training=cfg.DETECTOR.MODEL.RPN.POST_NMS_TOPN_TRAIN,
            testing=cfg.DETECTOR.MODEL.RPN.POST_NMS_TOPN_TEST,
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.DETECTOR.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.DETECTOR.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.DETECTOR.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.DETECTOR.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.DETECTOR.MODEL.RPN.NMS_THRESH,
        )
        if cfg.DETECTOR.MODEL.ROI_HEAD.USE_RES5:
            box_head = res5_head
        else:
            # trans head
            if cfg.REID_HEAD.L2P.DET_L2P:
                box_head = TransformerHeadL2PSimBox(
                    cfg=cfg.REID_HEAD,
                    kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
                )
            else:
                box_head = TransformerHeadSimBox(
                    cfg=cfg.REID_HEAD,
                    kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
                )
        reid_head = TransformerHeadL2PSimReid(
            cfg=cfg.REID_HEAD,
            kernel_size=cfg.REID_HEAD.CONV_KERNEL_SIZE,
        )

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        box_predictor = (
            FastRCNNPredictor(2048, 2)
            if cfg.DETECTOR.MODEL.ROI_HEAD.USE_RES5
            else FastRCNNPredictor(cfg.REID_HEAD.DIM_MODEL, 2)
        )
        roi_heads = TransROIHeadsL2PSim(
            cfg=cfg,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.DETECTOR.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.DETECTOR.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.DETECTOR.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.DETECTOR.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )

        if self.training:
            transform = GeneralizedRCNNTransform(
                min_size=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )
        else:
            transform = GeneralizedRCNNTransform(
                min_size=cfg.INPUT.MIN_SIZE_TEST,
                max_size=cfg.INPUT.MAX_SIZE_TEST,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )  # only to use post-process

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        lw_cfg = cfg.DETECTOR.LOSS.LOSS_WEIGHTS
        self.lw_rpn_reg = lw_cfg.LW_RPN_REG
        self.lw_rpn_cls = lw_cfg.LW_RPN_CLS
        self.lw_rcnn_reg = lw_cfg.LW_RCNN_REG
        self.lw_rcnn_cls = lw_cfg.LW_RCNN_CLS
        self.lw_rcnn_reid = cfg.REID_HEAD.LOSS.LOSS_WEIGHTS.OIM
        self.lw_l2p = cfg.REID_HEAD.LOSS.LOSS_WEIGHTS.L2P

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        self.vis_period = cfg.VIS_PERIOD

    def forward(self, input_list):
        res = super().forward(input_list)
        if self.training:
            for lk in res.keys():
                if "loss_l2p" in lk:
                    res[lk] *= self.lw_l2p
        return res

    def freeze_params(self):
        exclude_name = ["roi_heads"]

        def is_exluced(pn):
            exc = False
            for en in exclude_name:
                if en in pn:
                    exc = True
                    break
            return exc

        for name, param in self.named_parameters():
            if not is_exluced(name):
                param.requires_grad = False
        self.roi_heads.freeze_params()

    def inference(self, input_list):
        if "query" in input_list[0]:
            input_batches = self.preproxess_input([qd["query"] for qd in input_list])
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
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(
                features, boxes, images.image_sizes
            )
            box_features_2nd = self.roi_heads.box_head_2nd(box_features)[0]
            if "after_trans" in box_features_2nd:
                out_name = "after_trans"
            else:
                out_name = "feat_res5"
            box_embeddings_2nd = (
                box_features_2nd[out_name].squeeze(-1).squeeze(-1)
            )  # not l2 normed
            box_embeddings_2nd = self.roi_heads.bn_neck(box_embeddings_2nd)
            box_embeddings_2nd = F.normalize(box_embeddings_2nd, dim=1)
            embeddings = box_embeddings_2nd.cpu()  # .split(1, 0)
            result_list = input_list.copy()
            for bi, feat in enumerate(embeddings):
                result_list[bi]["query"]["feat"] = feat.view(-1)
            return result_list
        else:
            # gallery
            return super().inference(input_list)


class TransROIHeadsL2PSim(TransROIHeadsSim):
    def __init__(self, cfg, reid_head, *args, **kwargs):
        super().__init__(cfg, reid_head, *args, **kwargs)
        if isinstance(self.box_head_1st, TransformerHeadL2PSimBox):
            self.det_l2p = True
        else:
            self.det_l2p = False

    def freeze_params(self):
        exclude_name = ["box_head_2nd"]

        def is_exluced(pn):
            exc = False
            for en in exclude_name:
                if en in pn:
                    exc = True
                    break
            return exc

        for name, param in self.named_parameters():
            if not is_exluced(name):
                param.requires_grad = False

    def forward(self, features, boxes, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            boxes (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        gt_det_2nd = None
        feats_reid_2nd = None
        targets_reid_2nd = None

        if self.training:
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_1st,
                    self.bg_iou_thresh_1st,
                    allow_low_quality_matches=False,
                )
            (
                boxes,
                _,
                box_pid_labels_1st,
                box_reg_targets_1st,
            ) = self.select_training_samples(
                boxes, targets
            )  # 2000 input from rpn

        # ------------------- The first stage ------------------ # trans feat head + faster_rcnn_predictor
        box_features_1st = self.box_roi_pool(features, boxes, image_shapes)
        if self.det_l2p:
            box_features_1st, det_l2p_l = self.box_head_1st(box_features_1st)
        else:
            box_features_1st = self.box_head_1st(box_features_1st)
            det_l2p_l = None
        if "after_trans" in box_features_1st:
            out_name = "after_trans"
        else:
            out_name = "feat_res5"
        box_cls_scores_1st, box_regs_1st = self.box_predictor_1st(
            box_features_1st[out_name]
        )

        if self.training:
            boxes = self.get_boxes(box_regs_1st, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_2nd,
                    self.bg_iou_thresh_2nd,
                    allow_low_quality_matches=False,
                )
            (
                boxes,
                _,
                box_pid_labels_2nd,
                box_reg_targets_2nd,
            ) = self.select_training_samples(boxes, targets)
        else:
            orig_thresh = self.nms_thresh  # 0.4
            self.nms_thresh = self.nms_thresh_1st
            boxes, scores, _ = self.postprocess_proposals(
                box_cls_scores_1st, box_regs_1st, boxes, image_shapes
            )
            self.nms_thresh = orig_thresh

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes_i = torch.zeros(0, 4)
            labels_i = torch.zeros(0)
            scores_i = torch.zeros(0)
            embeddings_i = torch.zeros(0, 256)
            return [
                dict(
                    boxes=boxes_i,
                    labels=labels_i,
                    scores=scores_i,
                    embeddings=embeddings_i,
                )
                for _ in range(len(boxes))
            ], []

        # --------------------- The second stage -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features, l2p_l = self.box_head_2nd(box_features)
        if "after_trans" in box_features:
            out_name = "after_trans"
        else:
            out_name = "feat_res5"
        box_embeddings_2nd = box_features[out_name].squeeze(-1).squeeze(-1)

        # not l2 normed
        box_embeddings_2nd = self.bn_neck(box_embeddings_2nd)
        box_embeddings_2nd = F.normalize(box_embeddings_2nd, dim=1)
        # embeddings already l2 normed
        del box_features

        result, losses = [], {}
        if self.training:
            box_labels_1st = [y.clamp(0, 1) for y in box_pid_labels_1st]
            losses = detection_losses(
                box_cls_scores_1st,
                box_regs_1st,
                box_labels_1st,
                box_reg_targets_1st,
                None,
                None,
                None,
                None,
            )

            (
                loss_rcnn_reid_2nd,
                feats_reid_2nd,
                targets_reid_2nd,
                valid_inds,
            ) = self.reid_loss_2nd(box_embeddings_2nd, box_pid_labels_2nd)
            losses.update(loss_rcnn_reid_2nd=loss_rcnn_reid_2nd)
            losses.update(l2p_l)
            if det_l2p_l is not None:
                losses.update(det_l2p_l)
            # for vis only
            valid_inds_per_img = torch.split(
                valid_inds, [y.shape[0] for y in box_pid_labels_2nd]
            )
            valid_boxes = [b[vi] for b, vi in zip(boxes, valid_inds_per_img)]
            pid_labels_per_img = [
                pid[vi] for pid, vi in zip(box_pid_labels_2nd, valid_inds_per_img)
            ]
            for i in range(len(boxes)):
                result.append(
                    dict(
                        boxes=valid_boxes[i],
                        labels=pid_labels_per_img[i],
                    )
                )

        else:
            embeddings_2nd = torch.split(
                box_embeddings_2nd, [bx.shape[0] for bx in boxes]
            )
            num_images = len(boxes)
            for i in range(num_images):
                embeddings = embeddings_2nd[i]
                result.append(
                    dict(
                        boxes=boxes[i],
                        scores=scores[i],
                        embeddings=embeddings,
                    )
                )
                num_images = len(boxes)
                for i in range(num_images):
                    embeddings = embeddings_2nd[i]
                    result.append(
                        dict(
                            boxes=boxes[i],
                            scores=scores[i],
                            embeddings=embeddings,
                        )
                    )

        return (
            result,
            losses,
            feats_reid_2nd,
            targets_reid_2nd,
        )
