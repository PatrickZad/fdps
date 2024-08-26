from psd2.modeling.meta_arch import GeneralizedRCNN
from psd2.structures.boxes import Boxes
from ..build import META_ARCH_REGISTRY
import torch
from psd2.structures import ImageList, Instances
from psd2.utils.events import get_event_storage
from psd2.utils.visualizer import Visualizer, mlvl_pca_feat
from psd2.config import configurable
from psd2.modeling.poolers import ROIPooler
from psd2.layers import Conv2d, get_norm
from psd2.layers.mem_matching_losses import build_loss_layer
import torch.nn as nn
from psd2.layers.pooling import *
from psd2.modeling.reid_heads.id_assign import build_id_assigner
from psd2.modeling import build_backbone
from psd2.modeling.reid_heads.box_augmentation import build_box_augmentor
import numpy as np
import cv2
import torch.utils.checkpoint as checkpoint
import math
import random
from functools import reduce
import torch.nn.functional as F
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
import copy
from .ti_rcnn_baseline import TiRCNN_C4Side
@META_ARCH_REGISTRY.register()
class TiRCNN_Coat(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        reid_pooler,
        reid_loss,
        pfeat_pooling,
        train_task,
        pid_assigner,
        box_augmentor,
        train_with_det,
        train_with_nms,
        cws,
        use_checkpoint,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.reid_pooler = reid_pooler
        self.reid_loss = reid_loss
        self.pfeat_pooling = pfeat_pooling
        self.train_task = train_task
        self.pid_asigner = pid_assigner
        self.box_augmentor = box_augmentor
        self.train_with_nms = train_with_nms
        self.train_with_det = train_with_det
        self.cws = cws
        self.use_checkpoint = use_checkpoint

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        coat_cfg=cfg.REID_HEAD.COAT
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=1024,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=coat_cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=coat_cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
        )
        post_nms_top_n = dict(
            training=coat_cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=coat_cfg.MODEL.RPN.POST_NMS_TOPN_TEST
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=coat_cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=coat_cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=coat_cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=coat_cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=coat_cfg.MODEL.RPN.NMS_THRESH,
        )
        box_head = TransformerHead(
            cfg=coat_cfg,
            trans_names=coat_cfg.MODEL.TRANSFORMER.NAMES_1ST,
            kernel_size=coat_cfg.MODEL.TRANSFORMER.KERNEL_SIZE_1ST,
            use_feature_mask=coat_cfg.MODEL.TRANSFORMER.USE_MASK_1ST,
        )
        box_head_2nd = TransformerHead(
            cfg=coat_cfg,
            trans_names=coat_cfg.MODEL.TRANSFORMER.NAMES_2ND,
            kernel_size=coat_cfg.MODEL.TRANSFORMER.KERNEL_SIZE_2ND,
            use_feature_mask=coat_cfg.MODEL.TRANSFORMER.USE_MASK_2ND,
        )
        box_head_3rd = TransformerHead(
            cfg=coat_cfg,
            trans_names=coat_cfg.MODEL.TRANSFORMER.NAMES_3RD,
            kernel_size=coat_cfg.MODEL.TRANSFORMER.KERNEL_SIZE_3RD,
            use_feature_mask=coat_cfg.MODEL.TRANSFORMER.USE_MASK_3RD,
        )

        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["res4"], output_size=(16,8), sampling_ratio=2
        )
        box_predictor = FastRCNNPredictor(2048, 2)
        roi_heads = CascadedROIHeads(
            cfg=coat_cfg,
            # Cascade Transformer Head
            faster_rcnn_predictor=faster_rcnn_predictor,
            box_head_2nd=box_head_2nd,
            box_head_3rd=box_head_3rd,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=coat_cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=coat_cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=coat_cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=coat_cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=coat_cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=coat_cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=coat_cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )
        ret={
            "backbone": backbone,
            "proposal_generator": rpn,
            "roi_heads": roi_heads,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }
        reid_cfg = cfg.REID_HEAD
        reid_pooler_cfg = reid_cfg.ROI_POOLER
        ret["reid_pooler"] = ROIPooler(
            output_size=reid_pooler_cfg.POOLER_RESOLUTION,
            scales=reid_pooler_cfg.SCALES,
            sampling_ratio=reid_pooler_cfg.SAMP_RATIO,
            pooler_type=reid_pooler_cfg.TYPE,
            output_channels=1024,  #  compatible with deformRoiPool
        )
        loss_cfg = reid_cfg.LOSS
        loss_layer = build_loss_layer(loss_cfg, reid_cfg.PERSON_FEATURE.DIM)
        ret["reid_loss"] = loss_layer
        pool_layer = reid_cfg.POOL_LAYER
        if pool_layer == "fastavgpool":
            pfeat_pooling = FastGlobalAvgPool2d()
        elif pool_layer == "avgpool":
            pfeat_pooling = nn.AdaptiveAvgPool2d(1)
        elif pool_layer == "maxpool":
            pfeat_pooling = nn.AdaptiveMaxPool2d(1)
        elif pool_layer == "gempoolP":
            pfeat_pooling = GeneralizedMeanPoolingP()  # this one
        elif pool_layer == "gempool":
            pfeat_pooling = GeneralizedMeanPooling()
        elif pool_layer == "avgmaxpool":
            pfeat_pooling = AdaptiveAvgMaxPool2d()
        elif pool_layer == "clipavgpool":
            pfeat_pooling = ClipGlobalAvgPool2d()
        elif pool_layer == "identity":
            pfeat_pooling = nn.Identity()
        elif pool_layer == "flatten":
            pfeat_pooling = Flatten()
        elif pool_layer == "linear":
            pfeat_pooling = FlattenLinear(
                reid_pooler_cfg.POOLER_RESOLUTION[0]
                * reid_pooler_cfg.POOLER_RESOLUTION[1]
                * reid_cfg.PERSON_FEATURE.DIM,
                reid_cfg.PERSON_FEATURE.DIM,
            )
        else:
            raise KeyError(f"{pool_layer} is not supported!")
        ret["pfeat_pooling"] = pfeat_pooling
        ret["train_task"] = "ps" if reid_cfg.TRAIN_REID else "det"
        ret["pid_assigner"] = build_id_assigner(reid_cfg.ID_ASSIGN)
        ret["cws"] = cfg.CWS
        ret["train_with_nms"] = reid_cfg.TRAIN_WITH_NMS
        ret["train_with_det"] = reid_cfg.TRAIN_WITH_DET
        ret["box_augmentor"] = (
            build_box_augmentor(reid_cfg.BOX_AUGMENTATION)
            if reid_cfg.BOX_AUGMENTATION.ENABLE
            else None
        )
        ret["use_checkpoint"] = reid_cfg.USE_CHECKPOINT
        return ret

    def inf_query(self, input_list):
        images, gt_instances = self.preprocess_input(qd["query"] for qd in input_list)
        det_bk_features = self.backbone(images.tensor)
        reid_bk_feats = self.get_reid_backbone_features(det_bk_features, images)
        del det_bk_features
        q_boxes = [gti.gt_boxes.tensor for gti in gt_instances]
        q_featmaps = self.get_reid_person_features(reid_bk_feats, q_boxes)
        del reid_bk_feats
        q_embs = self.get_reid_embed(q_featmaps)
        del q_featmaps
        for bi, feat in enumerate(q_embs):
            input_list[bi]["query"]["feat"] = feat
        return input_list

    def get_reid_embed(self, p_feat_maps):
        raise NotImplementedError

    def forward_det(self, image_list, features, gt_instances):
        """
        return det feat maps and predictions
        """
        if self.training and self.train_task == "det":
            targets=[]
            for inst in gt_instances:
                targets.append({"boxes":inst.gt_boxes.tensor,"labels":inst.gt_classes})

            proposals, proposal_losses = self.proposal_generator(
                image_list, features, targets
            )
            det_pred_instances, det_losses = self.roi_heads(
                features, proposals,image_list.image_sizes, targets
            )
            det_pred_instances=det_pred_instances[-1]
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    outputs = {"pred_boxes": [], "pred_scores": []}
                    for pi, pred_img in enumerate(det_pred_instances):
                        boxes = pred_img.pred_boxes.tensor
                        scores = pred_img.scores
                        outputs["pred_boxes"].append(boxes)
                        outputs["pred_scores"].append(scores.unsqueeze(1))
                    for gti in gt_instances:
                        gt_id = gti.gt_classes
                        gt_id[gt_id == 1] = -1
                        gt_id[gt_id > 1] -= 2
                    self.visualize_training(
                        (image_list, gt_instances),
                        features[list(features.keys())[-1]],
                        outputs,
                    )
            losses = {}
            losses.update(det_losses)
            losses.update(proposal_losses)
            return losses
        elif self.training and self.train_task == "ps":
            if not self.train_with_det:
                return {
                    "pred_boxes": [],
                    "pred_scores": [],
                    "pred_pids": [],
                }
            if self.proposal_generator.training:
                self.proposal_generator.eval()
            if self.roi_heads.training:
                self.roi_heads.eval()
            proposals, _ = self.proposal_generator(image_list, features, None)
            if self.train_with_nms:
                if self.pid_asigner is None:
                    det_pred_instances = self.roi_heads.forward_labeling(
                        image_list, features, proposals, gt_instances, True, False
                    )
                    for pred in det_pred_instances:
                        pred_pids = pred.pred_classes
                        pred_pids[pred_pids == 1] = -2
                        pred_pids[pred_pids == 0] = -1
                        pred_pids[pred_pids > 1] -= 2
                else:
                    det_pred_instances, _ = self.roi_heads(
                        image_list, features, proposals, None
                    )
            else:
                if self.pid_asigner is None:
                    det_pred_instances = self.roi_heads.forward_labeling(
                        image_list, features, proposals, gt_instances, False, False
                    )
                    for pred in det_pred_instances:
                        pred_pids = pred.pred_classes
                        pred_pids[pred_pids == 1] = -2
                        pred_pids[pred_pids == 0] = -1
                        pred_pids[pred_pids > 1] -= 2
                else:
                    det_pred_instances = self.roi_heads.inference_without_nms(
                        image_list, features, proposals, None
                    )
            det_outputs = {"pred_boxes": [], "pred_scores": [], "pred_pids": []}
            for pred_img in det_pred_instances:
                boxes = pred_img.pred_boxes.tensor
                scores = pred_img.scores
                pids = pred_img.pred_classes
                det_outputs["pred_boxes"].append(boxes)
                det_outputs["pred_scores"].append(scores)
                det_outputs["pred_pids"].append(pids)
            # det_outputs["pred_boxes"] = torch.stack(det_outputs["pred_boxes"], dim=0)
            # det_outputs["pred_scores"] = torch.stack(det_outputs["pred_scores"], dim=0)
            return det_outputs
        else:
            proposals, _ = self.proposal_generator(image_list, features, None)
            det_pred_instances, _ = self.roi_heads(
                 features, proposals,image_list.image_sizes, None
            )
            det_pred_instances=det_pred_instances[-1]
            det_outputs = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pi, pred_img in enumerate(det_pred_instances):
                boxes = pred_img.pred_boxes.tensor
                scores = pred_img.scores
                det_outputs["pred_boxes"].append(boxes)
                det_outputs["pred_scores"].append(scores.unsqueeze(1))
                det_outputs["reid_feats"].append(boxes.clone())  # dumm impl
            return det_outputs

    def get_ps_pos_samples(self, det_pred, gts):
        if self.train_with_det:
            if self.pid_asigner is None:
                pbox_ids = det_pred.pop("pred_pids")
            else:
                pbox_ids = self.pid_asigner.assign(
                    det_pred["pred_boxes"],
                    [pi.view(pi.shape[0]) for pi in det_pred["pred_scores"]],
                    [gti.gt_boxes.tensor for gti in gts],
                    [gti.gt_classes for gti in gts],
                )
            det_pos_boxes, det_pos_ids, det_pos_scores = [], [], []
            gt_pos_boxes, gt_pos_ids, gt_pos_scores = [], [], []
            for boxes_i, ids_i, scores_i, gts_i in zip(
                det_pred["pred_boxes"], pbox_ids, det_pred["pred_scores"], gts
            ):
                keep = ids_i > -2
                keep_boxes = boxes_i[keep]
                keep_ids = ids_i[keep]
                det_pos_boxes.append(keep_boxes)
                det_pos_ids.append(keep_ids)
                det_pos_scores.append(scores_i[keep].unsqueeze(1))
                # append gt
                gt_pos_boxes.append(gts_i.gt_boxes.tensor)
                gt_pos_ids.append(gts_i.gt_classes)
                gt_pos_scores.append(
                    torch.ones_like(gts_i.gt_classes, dtype=scores_i.dtype).unsqueeze(1)
                )  # for vis only
            if self.box_augmentor is not None:
                pos_boxes, pos_ids = self.box_augmentor.augment_boxes(
                    gt_pos_boxes,
                    gt_pos_ids,
                    det_pos_boxes,
                    det_pos_ids,
                    [gti.image_size for gti in gts],
                )  # det appeded
                pos_scores = []
                for pi in range(len(pos_boxes)):
                    num_augs = pos_boxes[pi].shape[0] - det_pos_boxes[pi].shape[0]
                    num_append_gt = (
                        gt_pos_ids[pi].shape[0] if self.box_augmentor.append_gt else 0
                    )
                    pos_scores.append(
                        torch.cat(
                            [
                                torch.ones(
                                    num_augs,
                                    1,
                                    dtype=gt_pos_boxes[pi].dtype,
                                    device=gt_pos_boxes[pi].device,
                                ),
                                torch.ones(
                                    num_append_gt,
                                    1,
                                    dtype=gt_pos_boxes[pi].dtype,
                                    device=gt_pos_boxes[pi].device,
                                ),
                                det_pos_scores[pi],
                            ],
                            dim=0,
                        )
                    )
            else:
                pos_boxes, pos_ids, pos_scores = [], [], []
                for pi in range(len(gt_pos_boxes)):
                    pos_boxes.append(
                        torch.cat([det_pos_boxes[pi], gt_pos_boxes[pi]], dim=0)
                    )
                    pos_ids.append(torch.cat([det_pos_ids[pi], gt_pos_ids[pi]], dim=0))
                    pos_scores.append(
                        torch.cat([det_pos_scores[pi], gt_pos_scores[pi]], dim=0)
                    )

        else:
            pos_boxes, pos_ids, pos_scores = [], [], []
            for gts_i in gts:
                # append gt
                pos_boxes.append(gts_i.gt_boxes.tensor)
                pos_ids.append(gts_i.gt_classes)
                pos_scores.append(
                    torch.ones_like(
                        gts_i.gt_classes, dtype=gts_i.gt_boxes.tensor.dtype
                    ).unsqueeze(1)
                )  # for vis only
            if self.box_augmentor is not None:
                pos_boxes, pos_ids = self.box_augmentor.augment_boxes(
                    pos_boxes,
                    pos_ids,
                    det_boxes=None,
                    det_pids=None,
                    img_sizes=[gti.image_size for gti in gts],
                )
                pos_scores = [torch.ones_like(pids).unsqueeze(1) for pids in pos_ids]
        return pos_boxes, pos_ids, pos_scores

    def forward_ps(self, det_pred, det_backbone_features, image_list, gts):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        if self.training:
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples(det_pred, gts)

            pos_featmaps = self.get_reid_person_features(
                reid_bk_feats, pos_boxes
            )  # nf x c x h x w

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    vis_outputs = {
                        "pred_boxes": pos_boxes,
                        "pred_scores": pos_scores,
                        "assign_ids": pos_ids,
                    }
                    self.visualize_training(
                        (image_list, gts),
                        reid_bk_feats,
                        vis_outputs,
                    )
                    self.visualize_training_ps(
                        image_list.tensor,
                        pos_featmaps.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(pos_featmaps)
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            return self.reid_loss(pos_embs, pos_ids, None)
        else:
            num_boxes_per_image = [bi.shape[0] for bi in det_pred["pred_boxes"]]
            p_featmaps = self.get_reid_person_features(
                reid_bk_feats, det_pred["pred_boxes"]
            )
            del reid_bk_feats
            p_embs = self.get_reid_embed(p_featmaps)
            del p_featmaps
            if self.cws:
                cat_scores = torch.cat(det_pred["pred_scores"], dim=0)
                p_embs *= cat_scores  # .unsqueeze(1)
            p_embs = torch.split(p_embs, num_boxes_per_image)
            return p_embs

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        raise NotImplementedError

    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        raise NotImplementedError

    def forward(self, input_list):
        if self.training:
            images, gt_instances = self.preprocess_input(input_list)
            if self.train_task == "det":
                det_bk_features = self.backbone(images.tensor)
                return self.forward_det(images, det_bk_features, gt_instances)  # losses
            else:
                if self.backbone.training:
                    self.backbone.eval()
                with torch.no_grad():
                    det_bk_features = self.backbone(images.tensor)
                    det_pred = self.forward_det(images, det_bk_features, gt_instances)
                # back to original pid
                for gti in gt_instances:
                    cur_pids = gti.gt_classes
                    gti.gt_classes[cur_pids == 0] = -1
                    gti.gt_classes[cur_pids > 0] -= 2
                return self.forward_ps(det_pred, det_bk_features, images, gt_instances)
        else:
            if self.train_task == "det":
                images, gt_instances = self.preprocess_input(input_list)
                det_bk_features = self.backbone(images.tensor)
                det_pred = self.forward_det(
                    images, det_bk_features, gt_instances
                )  # det eval only
                for pi, bi_boxes in enumerate(det_pred["pred_boxes"]):
                    org_boxes = _resize_boxes(
                        bi_boxes, images.image_sizes[pi], gt_instances[pi]._org_hw
                    )
                    det_pred["pred_boxes"][pi] = org_boxes
                return det_pred
            if "query" in input_list[0]:
                return self.inf_query(input_list)
            images, gt_instances = self.preprocess_input(input_list)
            det_bk_features = self.backbone(images.tensor)
            det_pred = self.forward_det(images, det_bk_features, gt_instances)
            reid_feats = self.forward_ps(
                det_pred, det_bk_features, images, gt_instances
            )
            del det_bk_features
            outputs = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pi, pfeats in enumerate(reid_feats):
                boxes = det_pred["pred_boxes"][pi]
                org_boxes = _resize_boxes(
                    boxes, images.image_sizes[pi], gt_instances[pi]._org_hw
                )
                scores = det_pred["pred_scores"][pi]
                outputs["pred_boxes"].append(org_boxes)
                outputs["pred_scores"].append(scores)
                outputs["reid_feats"].append(pfeats)
            return outputs

    def preprocess_input(self, input_list):
        images = []
        gt_instances = []
        for input_dict in input_list:
            inst_img = Instances((input_dict["height"], input_dict["width"]))
            inst_img.gt_boxes = Boxes(
                torch.tensor(
                    input_dict["boxes"],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            inst_img._file_name = input_dict["file_name"]
            inst_img._image_id = input_dict["image_id"]
            images.append(input_dict["image"].to(self.device))
            org_pids = torch.tensor(input_dict["ids"], device=self.device).clone()
            org_pids[org_pids > -1] += 2  # labeled people pid +=2, >1
            org_pids[org_pids == -1] = 1  # unlabeled people pid =1
            inst_img.gt_classes = org_pids.long()
            inst_img.gt_classes = org_pids.long()
            inst_img._org_hw = (input_dict["org_height"], input_dict["org_width"])
            inst_img.org_boxes = Boxes(torch.tensor(input_dict["org_boxes"]))
            gt_instances.append(inst_img.to(self.device))
        return (
            ImageList.from_tensors(images, self.backbone.size_divisibility),
            gt_instances,
        )

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

        def score_splits(score, threds):
            splits = []
            for i, v in enumerate(threds):
                if score >= threds[i]:
                    splits.append(i)
            return splits

        threds = [0.5]
        storage = get_event_storage()
        trans_t2img_t = lambda t: t.detach().cpu() * self.pixel_std.cpu().view(
            -1, 1, 1
        ) + self.pixel_mean.cpu().view(-1, 1, 1)
        img_t2rgb = lambda t: (t.permute(1, 2, 0) * 255).numpy()
        samples = batched_inputs[0]
        annos = []
        for inst in batched_inputs[1]:
            trans_id = inst.gt_classes
            annos.append(
                {
                    "file_name": inst._file_name,
                    "image_id": inst._image_id,
                    "boxes": inst.gt_boxes.tensor,  # xyxy abs
                    "ids": trans_id,
                }
            )
        bs = len(annos)
        level_pcas = mlvl_pca_feat(featmap[None])
        for bi in range(bs):
            img_norm_t = samples.tensor[bi].cpu()
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

            visualize_runs = [Visualizer(img_rgb.copy()) for i in range(len(threds))]
            boxes = batched_dets["pred_boxes"][bi].detach().cpu()
            # TODO check if topk needed
            scores = (
                batched_dets["pred_scores"][bi]
                .detach()
                .cpu()
                .squeeze(1)
                .numpy()
                .tolist()
            )
            for i, score in enumerate(scores):
                splits = score_splits(score, threds)
                if len(splits) == 0:
                    continue
                b_clr = COLORS[i % len(COLORS)]
                t_clr = T_COLORS_BG[b_clr]
                for split in splits:
                    visualize_runs[split].draw_box(boxes[i], edge_color=b_clr)
                    visualize_runs[split].draw_text(
                        "%.2f" % score,
                        boxes[i][2:],
                        horizontal_alignment="right",
                        color=t_clr,
                        bg_color=b_clr,
                    )
                    if "assign_ids" in batched_dets:
                        visualize_runs[split].draw_text(
                            str(batched_dets["assign_ids"][bi][i].item()),
                            boxes[i][:2],
                            horizontal_alignment="left",
                            color=t_clr,
                            bg_color=b_clr,
                        )

            rgb_run_vis = [vis.get_output().get_image() for vis in visualize_runs]
            rgb_run_vis = np.concatenate(rgb_run_vis, axis=1)
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

    @torch.no_grad()
    def visualize_training_ps(self, images, p_feat_maps, p_boxes, p_ids):
        storage = get_event_storage()
        trans_t2img_t = lambda t: t.detach().cpu() * self.pixel_std.cpu().view(
            -1, 1, 1
        ) + self.pixel_mean.cpu().view(-1, 1, 1)
        feat_map_size = p_feat_maps[0].shape[-2:]
        tg_size = [feat_map_size[0] * 16, feat_map_size[1] * 16]
        bs = len(p_boxes)
        for bi in range(bs):
            boxes_bi = p_boxes[bi].cpu()  # n x 4
            box_areas = (boxes_bi[:, 2] - boxes_bi[:, 0]) * boxes_bi[:, 3] - boxes_bi[
                :, 1
            ]
            sort_idxs = torch.argsort(box_areas, dim=0, descending=True)
            feats_bi = p_feat_maps[bi].cpu()  # n x roi_h x roi_w
            idxs = sort_idxs[: min(10, sort_idxs.shape[0])]
            img_rgb_t = trans_t2img_t(images[bi].cpu())  # 3 x h x w
            assigns_on_boxes = []
            for i in idxs:
                assign_on_box = _render_attn_on_box(
                    img_rgb_t * 255, boxes_bi[i], feats_bi[i], tgt_size=tg_size
                )
                assigns_on_boxes.append(assign_on_box)
            cat_assigns_on_boxes = torch.cat(assigns_on_boxes, dim=2)
            storage.put_image("img_{}/attn".format(bi), cat_assigns_on_boxes / 255.0)


def _resize_boxes(boxes, original_size, new_size):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def _render_attn_on_box(img_rgb_t, pbox_t, feat_box, tgt_size=(384, 192)):
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
        feat_box[None], size=tgt_size, mode="bilinear", align_corners=False
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

    return pbox_img_rgb_t


from psd2.modeling.backbone.resnet import (
    ResNet,
    BottleneckBlock,
)
import torch.nn.init as init
import torch.nn.functional as tF



@META_ARCH_REGISTRY.register()
class TiRCNN_Coat_C4Side(TiRCNN_Coat):
    @configurable
    def __init__(self, side_init, **kwargs) -> None:
        super().__init__(**kwargs)

        self.side_init = side_init
        self.alpha_res3 = nn.Parameter(torch.tensor(0.0))
        self.alpha_res4 = nn.Parameter(torch.tensor(0.0))
        # side reid network
        self.side_res3 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 4,
                    "stride_per_block": [2, 1, 1, 1],
                    "in_channels": 256,
                    "out_channels": 512,
                    "norm": "BN",
                    "bottleneck_channels": 128,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        self.side_res4 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 6,
                    "stride_per_block": [2, 1, 1, 1, 1, 1],
                    "in_channels": 512,
                    "out_channels": 1024,
                    "norm": "BN",
                    "bottleneck_channels": 256,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        # freeze det params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.proposal_generator.parameters():
            p.requires_grad_(False)
        for pn,p in self.roi_heads.named_parameters():
            if "box" in pn:
                p.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        coat_cfg=cfg.REID_HEAD.COAT
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=1024,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=coat_cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=coat_cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
        )
        post_nms_top_n = dict(
            training=coat_cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=coat_cfg.MODEL.RPN.POST_NMS_TOPN_TEST
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=coat_cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=coat_cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=coat_cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=coat_cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=coat_cfg.MODEL.RPN.NMS_THRESH,
        )
        box_head = TransformerHead(
            cfg=coat_cfg,
            trans_names=coat_cfg.MODEL.TRANSFORMER.NAMES_1ST,
            kernel_size=coat_cfg.MODEL.TRANSFORMER.KERNEL_SIZE_1ST,
            use_feature_mask=coat_cfg.MODEL.TRANSFORMER.USE_MASK_1ST,
        )
        box_head_2nd = TransformerHead(
            cfg=coat_cfg,
            trans_names=coat_cfg.MODEL.TRANSFORMER.NAMES_2ND,
            kernel_size=coat_cfg.MODEL.TRANSFORMER.KERNEL_SIZE_2ND,
            use_feature_mask=coat_cfg.MODEL.TRANSFORMER.USE_MASK_2ND,
        )
        box_head_3rd = TransformerHead(
            cfg=coat_cfg,
            trans_names=coat_cfg.MODEL.TRANSFORMER.NAMES_3RD,
            kernel_size=coat_cfg.MODEL.TRANSFORMER.KERNEL_SIZE_3RD,
            use_feature_mask=coat_cfg.MODEL.TRANSFORMER.USE_MASK_3RD,
        )

        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["res4"], output_size=(16,8), sampling_ratio=2
        )
        box_predictor = FastRCNNPredictor(2048, 2)
        bn_neck = nn.BatchNorm1d(cfg.REID_HEAD.PERSON_FEATURE.DIM)
        init.constant_(bn_neck.bias, 0.0)
        init.normal_(bn_neck.weight, std=0.01)
        bn_neck.bias.requires_grad_(False)
        
        ret={
            "backbone": backbone,
            "proposal_generator": rpn,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }
        reid_cfg = cfg.REID_HEAD
        reid_pooler_cfg = reid_cfg.ROI_POOLER
        ret["reid_pooler"] = ROIPooler(
            output_size=reid_pooler_cfg.POOLER_RESOLUTION,
            scales=reid_pooler_cfg.SCALES,
            sampling_ratio=reid_pooler_cfg.SAMP_RATIO,
            pooler_type=reid_pooler_cfg.TYPE,
            output_channels=1024,  #  compatible with deformRoiPool
        )
        loss_cfg = reid_cfg.LOSS
        loss_layer = build_loss_layer(loss_cfg, reid_cfg.PERSON_FEATURE.DIM)
        pool_layer = reid_cfg.POOL_LAYER
        if pool_layer == "fastavgpool":
            pfeat_pooling = FastGlobalAvgPool2d()
        elif pool_layer == "avgpool":
            pfeat_pooling = nn.AdaptiveAvgPool2d(1)
        elif pool_layer == "maxpool":
            pfeat_pooling = nn.AdaptiveMaxPool2d(1)
        elif pool_layer == "gempoolP":
            pfeat_pooling = GeneralizedMeanPoolingP()  # this one
        elif pool_layer == "gempool":
            pfeat_pooling = GeneralizedMeanPooling()
        elif pool_layer == "avgmaxpool":
            pfeat_pooling = AdaptiveAvgMaxPool2d()
        elif pool_layer == "clipavgpool":
            pfeat_pooling = ClipGlobalAvgPool2d()
        elif pool_layer == "identity":
            pfeat_pooling = nn.Identity()
        elif pool_layer == "flatten":
            pfeat_pooling = Flatten()
        elif pool_layer == "linear":
            pfeat_pooling = FlattenLinear(
                reid_pooler_cfg.POOLER_RESOLUTION[0]
                * reid_pooler_cfg.POOLER_RESOLUTION[1]
                * reid_cfg.PERSON_FEATURE.DIM,
                reid_cfg.PERSON_FEATURE.DIM,
            )
        else:
            raise KeyError(f"{pool_layer} is not supported!")
        ret["reid_loss"] = loss_layer
        ret["pfeat_pooling"] = pfeat_pooling
        roi_heads = (CascadedTipsHeads if pool_layer == "maxpool" else CascadedTipsHeadsPooling)(
            oim2=copy.deepcopy(loss_layer),
            oim3=copy.deepcopy(loss_layer),
            pooling2=copy.deepcopy(pfeat_pooling),
            pooling3=copy.deepcopy(pfeat_pooling),
            bn_neck2=copy.deepcopy(bn_neck),
            bn_neck3=copy.deepcopy(bn_neck),
            cfg=coat_cfg,
            # Cascade Transformer Head
            faster_rcnn_predictor=faster_rcnn_predictor,
            box_head_2nd=box_head_2nd,
            box_head_3rd=box_head_3rd,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=coat_cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=coat_cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=coat_cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=coat_cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=coat_cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=coat_cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=coat_cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )
        ret["roi_heads"]= roi_heads
        ret["train_task"] = "ps" if reid_cfg.TRAIN_REID else "det"
        ret["pid_assigner"] = build_id_assigner(reid_cfg.ID_ASSIGN)
        ret["cws"] = cfg.CWS
        ret["train_with_nms"] = reid_cfg.TRAIN_WITH_NMS
        ret["train_with_det"] = reid_cfg.TRAIN_WITH_DET
        ret["box_augmentor"] = (
            build_box_augmentor(reid_cfg.BOX_AUGMENTATION)
            if reid_cfg.BOX_AUGMENTATION.ENABLE
            else None
        )
        ret["side_init"] = cfg.REID_HEAD.INIT_WEIGHT
        ret["use_checkpoint"] = reid_cfg.USE_CHECKPOINT

        return ret

    def load_state_dict(self, *args, **kws):
        output = super().load_state_dict(*args, **kws)
        model_resume = True
        for param_k in output.missing_keys:
            if "side_res" in param_k:
                model_resume = False
                break
        if not model_resume and self.side_init != "":
            # model is not resuming from saved checkpoint
            # init side networks
            state_dict = _load_file(self.side_init)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            side_res_params = [{}, {}, {}]
            bn_neck_params = {}
            for si in range(3):
                res_name = "res{}".format(si + 3)
                for pn, pa in state_dict.items():
                    if res_name in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        side_res_params[si][pn.split(res_name)[1][1:]] = pa
                    elif "bn_neck" in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        bn_neck_params[pn.split("bn_neck")[1][1:]] = pa
            self.side_res3.load_state_dict(side_res_params[0])
            self.side_res4.load_state_dict(side_res_params[1])
            if len(bn_neck_params) > 0:
                self.bn_neck.load_state_dict(bn_neck_params)
        return output

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        reid_res3_feat = (
            checkpoint.checkpoint(self.side_res3, det_backbone_features["res2"])
            if self.use_checkpoint
            else self.side_res3(det_backbone_features["res2"])
        )
        alpha = torch.sigmoid(self.alpha_res3)
        reid_res3_feat = (
            alpha * det_backbone_features["res3"] + (1 - alpha) * reid_res3_feat
        )
        reid_res4_feat = (
            checkpoint.checkpoint(self.side_res4, reid_res3_feat)
            if self.use_checkpoint
            else self.side_res4(reid_res3_feat)
        )
        del reid_res3_feat
        alpha = torch.sigmoid(self.alpha_res4)
        reid_res4_feat = (
            alpha * det_backbone_features["res4"] + (1 - alpha) * reid_res4_feat
        )
        return reid_res4_feat
    def inf_query(self, input_list):
        images, gt_instances = self.preprocess_input(qd["query"] for qd in input_list)
        det_bk_features = self.backbone(images.tensor)
        reid_bk_feats = self.get_reid_backbone_features(det_bk_features, images)
        del det_bk_features
        q_boxes =[gti.gt_boxes.tensor for gti in gt_instances]
        box_reid_features = self.roi_heads.box_roi_pool({"res4":reid_bk_feats}, q_boxes, images.image_sizes)
        del reid_bk_feats
        box_embeddings_2nd=self.roi_heads.embedding_head_2nd(box_reid_features)
        box_embeddings_3rd=self.roi_heads.embedding_head_3rd(box_reid_features)
        q_embs=torch.cat([F.normalize(box_embeddings_2nd,dim=-1),F.normalize(box_embeddings_3rd,dim=-1)],dim=-1)
        # q_embs=F.normalize(q_embs,dim=-1)
        for bi, feat in enumerate(q_embs):
            input_list[bi]["query"]["feat"] = feat
        return input_list
    def forward_ps(self, det_backbone_features, image_list, gts):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        
        if self.training:
            targets=[]
            for inst in gts:
                targets.append({"boxes":inst.gt_boxes.tensor,"labels":inst.gt_classes})
            proposals, _ = self.proposal_generator(
                image_list, {"res4":det_backbone_features["res4"]}, targets
            )
            _,reid_loss=self.roi_heads({"res4":det_backbone_features["res4"]},reid_bk_feats,proposals,image_list.image_sizes, targets)
            return reid_loss
        else:
            proposals, _ = self.proposal_generator(image_list, {"res4":det_backbone_features["res4"]}, None)
            pred_instances, _ = self.roi_heads(
                 {"res4":det_backbone_features["res4"]},reid_bk_feats,proposals,image_list.image_sizes, None
            )
            pred_instances=pred_instances[-1]
            return pred_instances,[inst.reid_feats for inst in pred_instances]
    def forward(self, input_list):
        if self.training:
            images, gt_instances = self.preprocess_input(input_list)
            if self.train_task == "det":
                det_bk_features = self.backbone(images.tensor)
                return self.forward_det(images, det_bk_features, gt_instances)  # losses
            else:
                if self.backbone.training:
                    self.backbone.eval()
                det_bk_features = self.backbone(images.tensor)
                return self.forward_ps(det_bk_features, images, gt_instances)
        else:
            if self.train_task == "det":
                images, gt_instances = self.preprocess_input(input_list)
                det_bk_features = self.backbone(images.tensor)
                det_pred = self.forward_det(
                    images, det_bk_features, gt_instances
                )  # det eval only
                for pi, bi_boxes in enumerate(det_pred["pred_boxes"]):
                    org_boxes = _resize_boxes(
                        bi_boxes, images.image_sizes[pi], gt_instances[pi]._org_hw
                    )
                    det_pred["pred_boxes"][pi] = org_boxes
                return det_pred
            if "query" in input_list[0]:
                return self.inf_query(input_list)
            images, gt_instances = self.preprocess_input(input_list)
            det_bk_features = self.backbone(images.tensor)
            det_pred ,reid_feats= self.forward_ps(
                 det_bk_features, images, gt_instances
            )
            del det_bk_features
            outputs = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pi, pfeats in enumerate(reid_feats):
                boxes = det_pred[pi].pred_boxes.tensor
                org_boxes = _resize_boxes(
                    boxes, images.image_sizes[pi], gt_instances[pi]._org_hw
                )
                scores = det_pred[pi].scores
                outputs["pred_boxes"].append(org_boxes)
                outputs["pred_scores"].append(scores.unsqueeze(1))
                outputs["reid_feats"].append(pfeats)
            return outputs


from psd2.layers.metric_loss import TripletLoss

@META_ARCH_REGISTRY.register()
class TiRCNN_Coat_C4SideTrip1(TiRCNN_Coat_C4Side):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.roi_heads.triplet_loss=TripletLoss(0.3, "mean")

    
@META_ARCH_REGISTRY.register()
class TiRCNN_C4SideTrans(TiRCNN_C4Side):
    @configurable
    def __init__(self,side_trans,*args,**kws):
        super().__init__(*args,**kws)
        del self.side_res5
        self.side_trans=side_trans
    @classmethod
    def from_config(cls, cfg):
        ret =super().from_config(cfg)
        coat_cfg=cfg.REID_HEAD.COAT

        box_head_2nd = TransformerHeadPlain(
            cfg=coat_cfg,
            trans_names=coat_cfg.MODEL.TRANSFORMER.NAMES_2ND,
            kernel_size=coat_cfg.MODEL.TRANSFORMER.KERNEL_SIZE_2ND,
            use_feature_mask=coat_cfg.MODEL.TRANSFORMER.USE_MASK_2ND,
        )
        ret["side_trans"]=box_head_2nd
        return ret
    
    def load_state_dict(self, *args, **kws):
        output = super(TiRCNN_C4Side,self).load_state_dict(*args, **kws)
        model_resume = True
        for param_k in output.missing_keys:
            if "side_res" in param_k:
                model_resume = False
                break
        if not model_resume and self.side_init != "":
            # model is not resuming from saved checkpoint
            # init side networks
            state_dict = _load_file(self.side_init)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            side_res_params = [{}, {}, {}]
            bn_neck_params = {}
            for si in range(3):
                res_name = "res{}".format(si + 3)
                for pn, pa in state_dict.items():
                    if res_name in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        side_res_params[si][pn.split(res_name)[1][1:]] = pa
                    elif "bn_neck" in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        bn_neck_params[pn.split("bn_neck")[1][1:]] = pa
            self.side_res3.load_state_dict(side_res_params[0])
            self.side_res4.load_state_dict(side_res_params[1])
            if len(bn_neck_params) > 0:
                self.bn_neck.load_state_dict(bn_neck_params)
        return output


    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats = self.reid_pooler([reid_backbone_features], d2_boxes)
        roi_feats = (
            checkpoint.checkpoint(self.side_trans, roi_feats)
            if self.use_checkpoint
            else self.side_trans(roi_feats)
        )  # n x c x h x w
        return roi_feats

@META_ARCH_REGISTRY.register()
class TiRCNN_C4SideVit(TiRCNN_C4SideTrans):
    @classmethod
    def from_config(cls, cfg):
        ret =super(TiRCNN_C4SideTrans,cls).from_config(cfg)
        reid_head = TransformerHeadVit(in_planes=1024)
        ret["side_trans"]=reid_head
        return ret
    
    def load_state_dict(self, *args, **kws):
        output = super(TiRCNN_C4Side,self).load_state_dict(*args, **kws)
        model_resume = True
        for param_k in output.missing_keys:
            if "side_res" in param_k:
                model_resume = False
                break
        if not model_resume and self.side_init != "":
            # model is not resuming from saved checkpoint
            # init side networks
            state_dict = _load_file(self.side_init)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            side_res_params = [{}, {}, {}]
            bn_neck_params = {}
            for si in range(3):
                res_name = "res{}".format(si + 3)
                for pn, pa in state_dict.items():
                    if res_name in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        side_res_params[si][pn.split(res_name)[1][1:]] = pa
                    elif "bn_neck" in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        bn_neck_params[pn.split("bn_neck")[1][1:]] = pa
            self.side_res3.load_state_dict(side_res_params[0])
            self.side_res4.load_state_dict(side_res_params[1])
            if len(bn_neck_params) > 0:
                self.bn_neck.load_state_dict(bn_neck_params)
        return output


def _load_file(filename):
    from psd2.utils.file_io import PathManager
    import pickle

    if filename.endswith(".pkl"):
        with PathManager.open(filename, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        if "model" in data and "__author__" in data:
            # file is in Detectron2 model zoo format
            return data
        else:
            # assume file is from Caffe2 / Detectron1 model zoo
            if "blobs" in data:
                # Detection models have "blobs", but ImageNet models don't
                data = data["blobs"]
            data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
            return {
                "model": data,
                "__author__": "Caffe2",
                "matching_heuristics": True,
            }
    elif filename.endswith(".pyth"):
        # assume file is from pycls; no one else seems to use the ".pyth" extension
        with PathManager.open(filename, "rb") as f:
            data = torch.load(f)
        assert (
            "model_state" in data
        ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
        model_state = {
            k: v
            for k, v in data["model_state"].items()
            if not k.endswith("num_batches_tracked")
        }
        return {
            "model": model_state,
            "__author__": "pycls",
            "matching_heuristics": True,
        }

    loaded = torch.load(
        filename, map_location=torch.device("cpu")
    )  # load native pth checkpoint
    if "model" not in loaded:
        loaded = {"model": loaded}
    return loaded



from psd2.structures import Boxes, ImageList, Instances
from torchvision.models.detection.roi_heads import RoIHeads
from copy import deepcopy
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops
class CascadedROIHeads(RoIHeads): # Det only
    '''
    https://github.com/pytorch/vision/blob/master/torchvision/models/detection/roi_heads.py
    '''
    def __init__(
        self,
        cfg,
        faster_rcnn_predictor,
        box_head_2nd,
        box_head_3rd,
        *args,
        **kwargs
    ):
        super(CascadedROIHeads, self).__init__(*args, **kwargs)

        # ROI head
        self.use_diff_thresh=cfg.MODEL.ROI_HEAD.USE_DIFF_THRESH
        self.nms_thresh_1st = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_1ST
        self.nms_thresh_2nd = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_2ND
        self.nms_thresh_3rd = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_3RD
        self.fg_iou_thresh_1st = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN
        self.bg_iou_thresh_1st = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN
        self.fg_iou_thresh_2nd = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN_2ND
        self.bg_iou_thresh_2nd = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_2ND
        self.fg_iou_thresh_3rd = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN_3RD
        self.bg_iou_thresh_3rd = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_3RD

        # Regression head
        self.box_predictor_1st = faster_rcnn_predictor
        self.box_predictor_2nd = self.box_predictor
        self.box_predictor_3rd = deepcopy(self.box_predictor)

        # Transformer head
        self.box_head_1st = self.box_head
        self.box_head_2nd = box_head_2nd
        self.box_head_3rd = box_head_3rd

        # feature mask
        self.use_feature_mask = cfg.MODEL.USE_FEATURE_MASK
        self.feature_mask_size = cfg.MODEL.FEATURE_MASK_SIZE

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
        gt_det_3rd = None
        pred_instances=[]
        matches=[]

        if self.training:
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_1st,
                    self.bg_iou_thresh_1st,
                    allow_low_quality_matches=False)
            boxes, matched_idxs, box_pid_labels_1st, box_reg_targets_1st = self.select_training_samples(
                boxes, targets
            )
            matches.append(matched_idxs)

        # ------------------- The first stage ------------------ #
        box_features_1st = self.box_roi_pool(features, boxes, image_shapes)
        box_features_1st = self.box_head_1st(box_features_1st)
        box_cls_scores_1st, box_regs_1st = self.box_predictor_1st(box_features_1st["after_trans"])

        if self.training:
            boxes = self.get_boxes(box_regs_1st, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            pred_scores=torch.softmax(box_cls_scores_1st.detach(),dim=-1)
            len_boxes=[boxes_per_image.shape[0] for boxes_per_image in boxes]
            pred_scores=torch.split(pred_scores,len_boxes)
            cur_preds=[]
            for boxes_i,scores_i,img_s_i in zip(boxes,pred_scores,image_shapes):
                cur_pred_i=Instances(img_s_i)
                cur_pred_i.pred_boxes=Boxes(boxes_i)
                cur_pred_i.scores=scores_i[:,1]
                cur_preds.append(cur_pred_i)
            matches.append(matched_idxs)
            pred_instances.append(cur_preds)

            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_2nd,
                    self.bg_iou_thresh_2nd,
                    allow_low_quality_matches=False)
            boxes, matched_idxs, box_pid_labels_2nd, box_reg_targets_2nd = self.select_training_samples(boxes, targets)
        else:
            orig_thresh = self.nms_thresh # 0.4
            self.nms_thresh = self.nms_thresh_1st
            boxes, scores, _ = self.postprocess_proposals(
                box_cls_scores_1st, box_regs_1st, boxes, image_shapes
            )

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes = gt_det_2nd["boxes"] if gt_det_2nd else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det_2nd else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det_2nd else torch.zeros(0)
            cur_preds=[]
            for img_s_i in zip(image_shapes):
                cur_pred_i=Instances(img_s_i)
                cur_pred_i.pred_boxes=Boxes(boxes.clone())
                cur_pred_i.scores=scores.clone()
                cur_preds.append(cur_pred_i)
            pred_instances.append(cur_preds)
            pred_instances.append(cur_preds)
            pred_instances.append(cur_preds)

            return pred_instances, {}
        else:
            if not self.training:
                cur_preds=[]
                for boxes_i,scores_i,img_s_i in zip(boxes,scores,image_shapes):
                    cur_pred_i=Instances(img_s_i)
                    cur_pred_i.pred_boxes=Boxes(boxes_i)
                    cur_pred_i.scores=scores_i.detach()
                    cur_preds.append(cur_pred_i)
                pred_instances.append(cur_preds)

        # --------------------- The second stage -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features = self.box_head_2nd(box_features)
        box_cls_scores_2nd, box_regs_2nd = self.box_predictor_2nd(box_features["after_trans"])
        if box_cls_scores_2nd.dim() == 0:
            box_cls_scores_2nd = box_cls_scores_2nd.unsqueeze(0)

        if self.training:
            boxes = self.get_boxes(box_regs_2nd, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            pred_scores=torch.softmax(box_cls_scores_2nd.detach(),dim=-1)
            len_boxes=[boxes_per_image.shape[0] for boxes_per_image in boxes]
            pred_scores=torch.split(pred_scores,len_boxes)
            cur_preds=[]
            for boxes_i,scores_i,img_s_i in zip(boxes,pred_scores,image_shapes):
                cur_pred_i=Instances(img_s_i)
                cur_pred_i.pred_boxes=Boxes(boxes_i)
                cur_pred_i.scores=scores_i.detach()[:,1]
                cur_preds.append(cur_pred_i)
            matches.append(matched_idxs)
            pred_instances.append(cur_preds)
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_3rd,
                    self.bg_iou_thresh_3rd,
                    allow_low_quality_matches=False)
            boxes, matched_idxs, box_pid_labels_3rd, box_reg_targets_3rd = self.select_training_samples(boxes, targets)
        else:
            self.nms_thresh = self.nms_thresh_2nd
            boxes, scores, _ = self.postprocess_detections(
                    box_cls_scores_2nd, box_regs_2nd, boxes, image_shapes
                )

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0 :
            assert not self.training
            boxes = gt_det_3rd["boxes"] if gt_det_3rd else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det_3rd else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det_3rd else torch.zeros(0)
            cur_preds=[]
            for img_s_i in zip(image_shapes):
                cur_pred_i=Instances(img_s_i)
                cur_pred_i.pred_boxes=Boxes(boxes.clone())
                cur_pred_i.scores=scores.clone()
                cur_preds.append(cur_pred_i)
            pred_instances.append(cur_preds)
            pred_instances.append(cur_preds)
            return pred_instances, {}
        else:
            if not self.training:
                cur_preds=[]
                for boxes_i,scores_i,img_s_i in zip(boxes,scores,image_shapes):
                    cur_pred_i=Instances(img_s_i)
                    cur_pred_i.pred_boxes=Boxes(boxes_i)
                    cur_pred_i.scores=scores_i.detach()
                    cur_preds.append(cur_pred_i)
                pred_instances.append(cur_preds)

        # --------------------- The third stage -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)

        box_features = self.box_head_3rd(box_features)
        box_cls_scores_3rd, box_regs_3rd = self.box_predictor_3rd(box_features["after_trans"])

        if box_cls_scores_3rd.dim() == 0:
            box_cls_scores_3rd = box_cls_scores_3rd.unsqueeze(0)

        losses =  {}
        if self.training:
            boxes = self.get_boxes(box_regs_3rd, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            pred_scores=torch.softmax(box_cls_scores_3rd.detach(),dim=-1)
            len_boxes=[boxes_per_image.shape[0] for boxes_per_image in boxes]
            pred_scores=torch.split(pred_scores,len_boxes)
            cur_preds=[]
            for boxes_i,scores_i,img_s_i in zip(boxes,pred_scores,image_shapes):
                cur_pred_i=Instances(img_s_i)
                cur_pred_i.pred_boxes=Boxes(boxes_i)
                cur_pred_i.scores=scores_i[:,1]
                cur_preds.append(cur_pred_i)
            matches.append(matched_idxs)
            pred_instances.append(cur_preds)
            box_labels_1st = [y.clamp(0, 1) for y in box_pid_labels_1st]
            box_labels_2nd = [y.clamp(0, 1) for y in box_pid_labels_2nd]
            box_labels_3rd = [y.clamp(0, 1) for y in box_pid_labels_3rd]
            losses = detection_losses(
                box_cls_scores_1st,
                box_regs_1st,
                box_labels_1st,
                box_reg_targets_1st,
                box_cls_scores_2nd,
                box_regs_2nd,
                box_labels_2nd,
                box_reg_targets_2nd,
                box_cls_scores_3rd,
                box_regs_3rd,
                box_labels_3rd,
                box_reg_targets_3rd,
            )
            return pred_instances,losses
        else:
            self.nms_thresh = self.nms_thresh_3rd
            boxes, scores, labels = self.postprocess_detections(
                    box_cls_scores_3rd, box_regs_3rd, boxes, image_shapes
                )
            # set to original thresh after finishing postprocess
            self.nms_thresh = orig_thresh

            cur_preds=[]
            for boxes_i,scores_i,img_s_i in zip(boxes,scores,image_shapes):
                cur_pred_i=Instances(img_s_i)
                cur_pred_i.pred_boxes=Boxes(boxes_i)
                cur_pred_i.scores=scores_i.detach()
                cur_preds.append(cur_pred_i)
            pred_instances.append(cur_preds)

        return pred_instances,{}

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

class CascadedTipsHeads(CascadedROIHeads):
    def __init__(
        self,
        oim2,oim3,pooling2,pooling3,bn_neck2,bn_neck3,
        *args,
        **kwargs
    ):  
        super().__init__(*args,**kwargs)
        self.oim2=oim2 
        self.oim3=oim3
        self.pooling2=pooling2
        self.pooling3=pooling3
        self.bn_neck2=bn_neck2
        self.bn_neck3=bn_neck3
        self.reid_head_2nd=copy.deepcopy(self.box_head_2nd)
        self.reid_head_3rd=copy.deepcopy(self.box_head_3rd)
    def embedding_head_2nd(self,box_features):
        box_features=self.reid_head_2nd(box_features)["after_trans"]
        # emb=self.pooling2(box_features).reshape(*box_features.shape[:2])
        emb=box_features.reshape(*box_features.shape[:2])
        emb=self.bn_neck2(emb)
        return emb
    def embedding_head_3rd(self,box_features):
        box_features=self.reid_head_3rd(box_features)["after_trans"]
        # emb=self.pooling3(box_features).reshape(*box_features.shape[:2])
        emb=box_features.reshape(*box_features.shape[:2])
        emb=self.bn_neck3(emb)
        return emb
    def reid_loss_2nd(self,feats,ids):
        ids=torch.cat(ids,dim=0).clone()
        ids[ids==0]=-2
        ids[ids==1]=-1
        ids[ids>1]-=2
        loss={}
        if hasattr(self,"triplet_loss"):
            oim_lookup = self.oim2.lb_layer.lookup_table
            lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=ids.dtype, device=ids.device
            )
            feats1 = feats[ids > -1]
            feats2 = torch.cat([feats[ids > -1], oim_lookup], dim=0)
            if feats1.shape[0] < 1:
                loss["loss_triplet"] = torch.zeros(1, device=ids.device)
            else:
                trip = self.triplet_loss(
                    feats1,
                    feats2,
                    ids[ids > -1],
                    torch.cat([ids[ids > -1], lookup_ids], dim=0),
                    normalize_feature=True,
                )
                loss.update(trip)
        loss.update(self.oim2(feats,ids,None))
        return loss
    def reid_loss_3rd(self,feats,ids):
        ids=torch.cat(ids,dim=0).clone()
        ids[ids==0]=-2
        ids[ids==1]=-1
        ids[ids>1]-=2
        loss={}
        if hasattr(self,"triplet_loss"):
            oim_lookup = self.oim3.lb_layer.lookup_table
            lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=ids.dtype, device=ids.device
            )
            feats1 = feats[ids > -1]
            feats2 = torch.cat([feats[ids > -1], oim_lookup], dim=0)
            if feats1.shape[0] < 1:
                loss["loss_triplet"] = torch.zeros(1, device=ids.device)
            else:
                trip = self.triplet_loss(
                    feats1,
                    feats2,
                    ids[ids > -1],
                    torch.cat([ids[ids > -1], lookup_ids], dim=0),
                    normalize_feature=True,
                )
                loss.update(trip)
        loss.update(self.oim3(feats,ids,None))
        return loss
    
    def forward(self, features, features_reid,boxes, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            boxes (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        cws = True
        pred_instances=[]
        matches=[]

        if self.training:
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_1st,
                    self.bg_iou_thresh_1st,
                    allow_low_quality_matches=False)
            boxes, matched_idxs, box_pid_labels_1st, box_reg_targets_1st = self.select_training_samples(
                boxes, targets
            )
            matches.append(matched_idxs)

        # ------------------- The first stage ------------------ #
        box_features_1st = self.box_roi_pool(features, boxes, image_shapes)
        box_features_1st = self.box_head_1st(box_features_1st)
        box_cls_scores_1st, box_regs_1st = self.box_predictor_1st(box_features_1st["after_trans"])

        if self.training:
            boxes = self.get_boxes(box_regs_1st, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            pred_scores=torch.softmax(box_cls_scores_1st.detach(),dim=-1)
            len_boxes=[boxes_per_image.shape[0] for boxes_per_image in boxes]
            pred_scores=torch.split(pred_scores,len_boxes)
            cur_preds=[]
            for boxes_i,scores_i,img_s_i in zip(boxes,pred_scores,image_shapes):
                cur_pred_i=Instances(img_s_i)
                cur_pred_i.pred_boxes=Boxes(boxes_i)
                cur_pred_i.scores=scores_i[:,1]
                cur_preds.append(cur_pred_i)
            matches.append(matched_idxs)
            pred_instances.append(cur_preds)

            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_2nd,
                    self.bg_iou_thresh_2nd,
                    allow_low_quality_matches=False)
            boxes, matched_idxs, box_pid_labels_2nd, box_reg_targets_2nd = self.select_training_samples(boxes, targets)
        else:
            orig_thresh = self.nms_thresh # 0.4
            self.nms_thresh = self.nms_thresh_1st
            boxes, scores, _ = self.postprocess_proposals(
                box_cls_scores_1st, box_regs_1st, boxes, image_shapes
            )

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes = torch.zeros(0, 4)
            labels =  torch.zeros(0)
            scores =  torch.zeros(0)
            embeddings = torch.zeros(0, 4096)
            cur_preds=[]
            for img_s_i in zip(image_shapes):
                cur_pred_i=Instances(img_s_i)
                cur_pred_i.pred_boxes=Boxes(boxes.clone())
                cur_pred_i.scores=scores.clone()
                cur_pred_i.reid_efats=embeddings
                cur_preds.append(cur_pred_i)
            pred_instances.append(cur_preds)
            pred_instances.append(cur_preds)
            pred_instances.append(cur_preds)

            return pred_instances, {}
        else:
            if not self.training:
                cur_preds=[]
                for boxes_i,scores_i,img_s_i in zip(boxes,scores,image_shapes):
                    cur_pred_i=Instances(img_s_i)
                    cur_pred_i.pred_boxes=Boxes(boxes_i)
                    cur_pred_i.scores=scores_i.detach()
                    cur_preds.append(cur_pred_i)
                pred_instances.append(cur_preds)

        # --------------------- The second stage -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_reid_features = self.box_roi_pool({"res4":features_reid}, boxes, image_shapes)
        box_embeddings_2nd=self.embedding_head_2nd(box_reid_features)
        box_features = self.box_head_2nd(box_features)
        box_cls_scores_2nd, box_regs_2nd = self.box_predictor_2nd(box_features["after_trans"])
        
        if box_cls_scores_2nd.dim() == 0:
            box_cls_scores_2nd = box_cls_scores_2nd.unsqueeze(0)

        if self.training:
            boxes = self.get_boxes(box_regs_2nd, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            pred_scores=torch.softmax(box_cls_scores_2nd.detach(),dim=-1)
            len_boxes=[boxes_per_image.shape[0] for boxes_per_image in boxes]
            pred_scores=torch.split(pred_scores,len_boxes)
            cur_preds=[]
            for boxes_i,scores_i,img_s_i in zip(boxes,pred_scores,image_shapes):
                cur_pred_i=Instances(img_s_i)
                cur_pred_i.pred_boxes=Boxes(boxes_i)
                cur_pred_i.scores=scores_i.detach()[:,1]
                cur_preds.append(cur_pred_i)
            matches.append(matched_idxs)
            pred_instances.append(cur_preds)
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_3rd,
                    self.bg_iou_thresh_3rd,
                    allow_low_quality_matches=False)
            boxes, matched_idxs, box_pid_labels_3rd, box_reg_targets_3rd = self.select_training_samples(boxes, targets)
        else:
            self.nms_thresh = self.nms_thresh_2nd
            boxes, scores, _, _ = self.postprocess_boxes(
                    box_cls_scores_2nd,
                    box_regs_2nd,
                    box_embeddings_2nd,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=None,
                    cws=cws,
                )

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0 :
            assert not self.training
            boxes =  torch.zeros(0, 4)
            labels =  torch.zeros(0)
            scores =   torch.zeros(0)
            embeddings = torch.zeros(0, 4096)
            cur_preds=[]
            for img_s_i in zip(image_shapes):
                cur_pred_i=Instances(img_s_i)
                cur_pred_i.pred_boxes=Boxes(boxes.clone())
                cur_pred_i.scores=scores.clone()
                cur_pred_i.reid_feats=embeddings.clone()
                cur_preds.append(cur_pred_i)
            pred_instances.append(cur_preds)
            pred_instances.append(cur_preds)
            return pred_instances, {}
        else:
            if not self.training:
                cur_preds=[]
                for boxes_i,scores_i,img_s_i in zip(boxes,scores,image_shapes):
                    cur_pred_i=Instances(img_s_i)
                    cur_pred_i.pred_boxes=Boxes(boxes_i)
                    cur_pred_i.scores=scores_i.detach()
                    cur_preds.append(cur_pred_i)
                pred_instances.append(cur_preds)

        # --------------------- The third stage -------------------- #
        if not self.training:
            box_reid_features = self.box_roi_pool({"res4":features_reid}, boxes, image_shapes)
            box_embeddings_2nd=self.embedding_head_2nd(box_reid_features)
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_reid_features = self.box_roi_pool({"res4":features_reid}, boxes, image_shapes)
        box_embeddings_3rd=self.embedding_head_3rd(box_reid_features)
        box_features = self.box_head_3rd(box_features)
        box_cls_scores_3rd, box_regs_3rd = self.box_predictor_3rd(box_features["after_trans"])

        if box_cls_scores_3rd.dim() == 0:
            box_cls_scores_3rd = box_cls_scores_3rd.unsqueeze(0)

        losses =  {}
        if self.training:
            loss_reid_2nd = self.reid_loss_2nd(box_embeddings_2nd, box_pid_labels_2nd)
            loss_reid_3rd = self.reid_loss_3rd(box_embeddings_3rd, box_pid_labels_3rd)
            for k,v in loss_reid_2nd.items():
                if "loss" in k:
                    losses[k+"_2nd"]=v
            for k,v in loss_reid_3rd.items():
                if "loss" in k:
                    losses[k+"_3rd"]=v
            boxes = self.get_boxes(box_regs_3rd, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            pred_scores=torch.softmax(box_cls_scores_3rd.detach(),dim=-1)
            len_boxes=[boxes_per_image.shape[0] for boxes_per_image in boxes]
            pred_scores=torch.split(pred_scores,len_boxes)
            cur_preds=[]
            for boxes_i,scores_i,img_s_i in zip(boxes,pred_scores,image_shapes):
                cur_pred_i=Instances(img_s_i)
                cur_pred_i.pred_boxes=Boxes(boxes_i)
                cur_pred_i.scores=scores_i[:,1]
                cur_preds.append(cur_pred_i)
            matches.append(matched_idxs)
            pred_instances.append(cur_preds)
            return pred_instances,losses
        else:
            self.nms_thresh = self.nms_thresh_3rd
            box_embeddings=torch.cat([F.normalize(box_embeddings_2nd,dim=-1),F.normalize(box_embeddings_3rd,dim=-1)],dim=-1)
            boxes, scores, box_embeddings, labels = self.postprocess_boxes(
                    box_cls_scores_3rd,
                    box_regs_3rd,
                    box_embeddings,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=None,
                    cws=cws,
                )
            # set to original thresh after finishing postprocess
            self.nms_thresh = orig_thresh

            cur_preds=[]
            for boxes_i,scores_i,img_s_i,feats_i in zip(boxes,scores,image_shapes,box_embeddings):
                cur_pred_i=Instances(img_s_i)
                cur_pred_i.pred_boxes=Boxes(boxes_i)
                cur_pred_i.scores=scores_i.detach()
                cur_pred_i.reid_feats=feats_i # F.normalize(feats_i,dim=-1)
                cur_preds.append(cur_pred_i)
            pred_instances.append(cur_preds)

        return pred_instances,{}
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
            pred_scores = torch.cat(fcs) # fcs[0]
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
            # embeddings = embeddings.reshape(-1, self.embedding_head_2nd.dim)

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

class CascadedTipsHeadsPooling(CascadedTipsHeads):
    def __init__(
        self,
        *args,
        **kwargs
    ):  
        super().__init__(*args,**kwargs)
        self.reid_head_2nd.pooling_before=copy.deepcopy(self.pooling2)
        self.reid_head_2nd.pooling_after=copy.deepcopy(self.pooling2)
        self.reid_head_3rd.pooling_before=copy.deepcopy(self.pooling3)
        self.reid_head_3rd.pooling_after=copy.deepcopy(self.pooling3)


def detection_losses(
    box_cls_scores_1st,
    box_regs_1st,
    box_labels_1st,
    box_reg_targets_1st,
    box_cls_scores_2nd,
    box_regs_2nd,
    box_labels_2nd,
    box_reg_targets_2nd,
    box_cls_scores_3rd,
    box_regs_3rd,
    box_labels_3rd,
    box_reg_targets_3rd,
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
    box_labels_2nd = torch.cat(box_labels_2nd, dim=0)
    box_reg_targets_2nd = torch.cat(box_reg_targets_2nd, dim=0)
    loss_rcnn_cls_2nd =F.cross_entropy(box_cls_scores_2nd, box_labels_2nd)    # F.binary_cross_entropy_with_logits(box_cls_scores_2nd, box_labels_2nd.float())

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

    # --------------------- The third stage -------------------- #
    box_labels_3rd = torch.cat(box_labels_3rd, dim=0)
    box_reg_targets_3rd = torch.cat(box_reg_targets_3rd, dim=0)
    loss_rcnn_cls_3rd =F.cross_entropy(box_cls_scores_3rd, box_labels_3rd)  # F.binary_cross_entropy_with_logits(box_cls_scores_3rd, box_labels_3rd.float())

    sampled_pos_inds_subset = torch.nonzero(box_labels_3rd > 0).squeeze(1)
    labels_pos = box_labels_3rd[sampled_pos_inds_subset]
    N = box_cls_scores_3rd.size(0)
    box_regs_3rd = box_regs_3rd.reshape(N, -1, 4)

    loss_rcnn_reg_3rd = F.smooth_l1_loss(
        box_regs_3rd[sampled_pos_inds_subset, labels_pos],
        box_reg_targets_3rd[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_rcnn_reg_3rd = loss_rcnn_reg_3rd / box_labels_3rd.numel()

    return dict(
        loss_rcnn_cls_1st=loss_rcnn_cls_1st,
        loss_rcnn_reg_1st=loss_rcnn_reg_1st,
        loss_rcnn_cls_2nd=loss_rcnn_cls_2nd,
        loss_rcnn_reg_2nd=loss_rcnn_reg_2nd,
        loss_rcnn_cls_3rd=loss_rcnn_cls_3rd,
        loss_rcnn_reg_3rd=loss_rcnn_reg_3rd,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class TransformerHead(nn.Module):
    def __init__(
        self,
        cfg,
        trans_names, 
        kernel_size,
        use_feature_mask,
    ):
        super(TransformerHead, self).__init__()
        d_model = cfg.MODEL.TRANSFORMER.DIM_MODEL

        # Mask parameters
        self.use_feature_mask = use_feature_mask
        mask_shape = cfg.MODEL.MASK_SHAPE
        mask_size = cfg.MODEL.MASK_SIZE
        mask_mode = cfg.MODEL.MASK_MODE

        self.bypass_mask = exchange_patch(mask_shape, mask_size, mask_mode)
        self.get_mask_box = get_mask_box(mask_shape, mask_size, mask_mode)

        self.transformer_encoder = Transformers(
            cfg=cfg,
            trans_names=trans_names, 
            kernel_size=kernel_size,
            use_feature_mask=use_feature_mask,
        )
        self.conv0 = conv1x1(1024, 1024)
        self.conv1 = conv1x1(1024, d_model)
        self.conv2 = conv1x1(d_model, 2048)

    def forward(self, box_features):
        mask_box = self.get_mask_box(box_features)

        if self.use_feature_mask:
            skip_features = self.conv0(box_features)
            if self.training:
                skip_features = self.bypass_mask(skip_features)
        else:
            skip_features = box_features

        trans_features = {}

        if hasattr(self,"pooling_before"):
            trans_features["before_trans"] = self.pooling_before(skip_features)
        else:
            trans_features["before_trans"] = F.adaptive_max_pool2d(skip_features, 1)
        box_features = self.conv1(box_features)
        box_features = self.transformer_encoder((box_features,mask_box))
        box_features = self.conv2(box_features)
        if hasattr(self,"pooling_after"):
            trans_features["after_trans"] = self.pooling_after(box_features)
        else:
            trans_features["after_trans"] = F.adaptive_max_pool2d(box_features, 1)

        return trans_features

class TransformerHeadPlain(TransformerHead):
    def forward(self, box_features):
        mask_box = self.get_mask_box(box_features)

        if self.use_feature_mask:
            skip_features = self.conv0(box_features)
            if self.training:
                skip_features = self.bypass_mask(skip_features)
        else:
            skip_features = box_features
        box_features = self.conv1(box_features)
        box_features = self.transformer_encoder((box_features,mask_box))
        box_features = self.conv2(box_features)
        return box_features

class Transformers(nn.Module):
    def __init__(
        self,
        cfg,
        trans_names, 
        kernel_size,
        use_feature_mask,
    ):
        super(Transformers, self).__init__()
        d_model = cfg.MODEL.TRANSFORMER.DIM_MODEL
        self.feature_aug_type = cfg.MODEL.FEATURE_AUG_TYPE
        self.use_feature_mask = use_feature_mask

        # If no conv before transformer, we do not use scales
        if not cfg.MODEL.TRANSFORMER.USE_PATCH2VEC:
            trans_names = ['scale1']
            kernel_size = [(1,1)]

        self.trans_names = trans_names
        self.scale_size = len(self.trans_names)
        hidden = d_model//(2*self.scale_size)

        # kernel_size: (padding, stride)
        kernels = {
            (1,1): [(0,0),(1,1)],
            (3,3): [(1,1),(1,1)]
        }

        padding = []
        stride = []
        for ksize in kernel_size:
            ksize=tuple(ksize)
            if ksize not in [(1,1),(3,3)]:
                raise ValueError('Undefined kernel size.')
            padding.append(kernels[ksize][0])
            stride.append(kernels[ksize][1])

        self.use_output_layer = cfg.MODEL.TRANSFORMER.USE_OUTPUT_LAYER
        self.use_global_shortcut = cfg.MODEL.TRANSFORMER.USE_GLOBAL_SHORTCUT

        self.blocks = nn.ModuleDict()
        for tname, ksize, psize, ssize in zip(self.trans_names, kernel_size, padding, stride):
            transblock = Transformer(
                cfg, d_model//self.scale_size, ksize, psize, ssize, hidden, use_feature_mask
            )
            self.blocks[tname] = nn.Sequential(transblock)

        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mask_para = [cfg.MODEL.MASK_SHAPE, cfg.MODEL.MASK_SIZE, cfg.MODEL.MASK_MODE]

    def forward(self, inputs):
        trans_feat = []
        enc_feat, mask_box = inputs

        if self.training and self.use_feature_mask and self.feature_aug_type == 'exchange_patch':
            feature_mask = exchange_patch(self.mask_para[0], self.mask_para[1], self.mask_para[2])
            enc_feat = feature_mask(enc_feat)

        for tname, feat in zip(self.trans_names, torch.chunk(enc_feat, len(self.trans_names), dim=1)):
            feat = self.blocks[tname]((feat, mask_box))
            trans_feat.append(feat)

        trans_feat = torch.cat(trans_feat, 1)
        if self.use_output_layer:
            trans_feat = self.output_linear(trans_feat)
        if self.use_global_shortcut:
            trans_feat = enc_feat + trans_feat
        return trans_feat


class Transformer(nn.Module):
    def __init__(self, cfg, channel, kernel_size, padding, stride, hidden, use_feature_mask
        ):
        super(Transformer, self).__init__()
        self.k = kernel_size[0]
        stack_num = cfg.MODEL.TRANSFORMER.ENCODER_LAYERS
        num_head = cfg.MODEL.TRANSFORMER.N_HEAD
        dropout = cfg.MODEL.TRANSFORMER.DROPOUT
        output_size = (16,8) #(14,14)
        token_size = tuple(map(lambda x,y:x//y, output_size, stride))
        blocks = []
        self.transblock = TransformerBlock(token_size, hidden=hidden, num_head=num_head, dropout=dropout)
        for _ in range(stack_num):
            blocks.append(self.transblock)
        self.transformer = nn.Sequential(*blocks)
        self.patch2vec = nn.Conv2d(channel, hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.vec2patch = Vec2Patch(channel, hidden, output_size, kernel_size, stride, padding)
        self.use_local_shortcut = cfg.MODEL.TRANSFORMER.USE_LOCAL_SHORTCUT
        self.use_feature_mask = use_feature_mask
        self.feature_aug_type = cfg.MODEL.FEATURE_AUG_TYPE
        self.use_patch2vec = cfg.MODEL.TRANSFORMER.USE_PATCH2VEC

    def forward(self, inputs):
        enc_feat, mask_box = inputs
        b, c, h, w = enc_feat.size()

        trans_feat = self.patch2vec(enc_feat)

        _, c, h, w = trans_feat.size()
        trans_feat = trans_feat.view(b, c, -1).permute(0, 2, 1)

        # For 1x1 & 3x3 kernels, exchange tokens
        if self.training and self.use_feature_mask:
            if self.feature_aug_type == 'exchange_token':
                feature_mask = exchange_token()
                trans_feat = feature_mask(trans_feat, mask_box)
            elif self.feature_aug_type == 'cutout_patch':
                feature_mask = cutout_patch()
                trans_feat = feature_mask(trans_feat)
            elif self.feature_aug_type == 'erase_patch':
                feature_mask = erase_patch()
                trans_feat = feature_mask(trans_feat)
            elif self.feature_aug_type == 'mixup_patch':
                feature_mask = mixup_patch()
                trans_feat = feature_mask(trans_feat)

        if self.use_feature_mask:
            if self.feature_aug_type == 'jigsaw_patch':
                feature_mask = jigsaw_patch()
                trans_feat = feature_mask(trans_feat)
            elif self.feature_aug_type == 'jigsaw_token':
                feature_mask = jigsaw_token()
                trans_feat = feature_mask(trans_feat)

        trans_feat = self.transformer(trans_feat)
        trans_feat = self.vec2patch(trans_feat)
        if self.use_local_shortcut:
            trans_feat = enc_feat + trans_feat

        return trans_feat


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, tokensize, hidden=128, num_head=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(tokensize, d_model=hidden, head=num_head, p=dropout)
        self.ffn = FeedForward(hidden, p=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.norm1(x)
        x = x + self.dropout(self.attention(x))
        y = self.norm2(x)
        x = x + self.ffn(y)

        return x


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class Vec2Patch(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(Vec2Patch, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.to_patch = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        h, w = output_size

    def forward(self, x):
        feat = self.embedding(x)
        b, n, c = feat.size()
        feat = feat.permute(0, 2, 1)
        feat = self.to_patch(feat)

        return feat

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, tokensize, d_model, head, p=0.1):
        super().__init__()
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head
        self.h, self.w = tokensize

    def forward(self, x):
        b, n, c = x.size() 
        c_h = c // self.head
        key = self.key_embedding(x)
        query = self.query_embedding(x)
        value = self.value_embedding(x)
        key = key.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        query = query.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        value = value.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        att, _ = self.attention(query, key, value)
        att = att.permute(0, 2, 1, 3).contiguous().view(b, n, c)
        output = self.output_linear(att)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=p))

    def forward(self, x):
        x = self.conv(x)
        return x

class exchange_token:
    def __init__(self):
        pass

    def __call__(self, features, mask_box):
        b, hw, c = features.size()
        assert hw == 16*8
        new_idx, mask_x1, mask_x2, mask_y1, mask_y2 = mask_box
        features = features.view(b, 16, 8, c)
        features[:, mask_x1 : mask_x2, mask_y1 : mask_y2, :] = features[new_idx, mask_x1 : mask_x2, mask_y1 : mask_y2, :]
        features = features.view(b, hw, c)
        return features

class jigsaw_token:
    def __init__(self, shift=5, group=2, begin=1):
        self.shift = shift
        self.group = group
        self.begin = begin

    def __call__(self, features):
        batchsize = features.size(0)
        dim = features.size(2)

        num_tokens = features.size(1)
        if num_tokens == 196:
            self.group = 2
        elif num_tokens == 25:
            self.group = 5
        else:
            raise Exception("Jigsaw - Unwanted number of tokens")

        # Shift Operation
        feature_random = torch.cat([features[:, self.begin-1+self.shift:, :], features[:, self.begin-1:self.begin-1+self.shift, :]], dim=1)
        x = feature_random

        # Patch Shuffle Operation
        try:
            x = x.view(batchsize, self.group, -1, dim)
        except:
            raise Exception("Jigsaw - Unwanted number of groups")

        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, dim)

        return x

class get_mask_box:
    def __init__(self, shape='stripe', mask_size=2, mode='random_direct'):
        self.shape = shape
        self.mask_size = mask_size
        self.mode = mode

    def __call__(self, features):
        # Stripe mask
        if self.shape == 'stripe':
            if self.mode == 'horizontal':
                mask_box = self.hstripe(features, self.mask_size)
            elif self.mode == 'vertical':
                mask_box = self.vstripe(features, self.mask_size)
            elif self.mode == 'random_direction':
                if random.random() < 0.5:
                    mask_box = self.hstripe(features, self.mask_size)
                else:
                    mask_box = self.vstripe(features, self.mask_size)
            else:
                raise Exception("Unknown stripe mask mode name")
        # Square mask
        elif self.shape == 'square':
            if self.mode == 'random_size':
                self.mask_size = 4 if random.random() < 0.5 else 5
            mask_box = self.square(features, self.mask_size)
        # Random stripe/square mask
        elif self.shape == 'random':
            random_num = random.random()
            if random_num < 0.25:
                mask_box = self.hstripe(features, 2)
            elif random_num < 0.5 and random_num >= 0.25:
                mask_box = self.vstripe(features, 2)
            elif random_num < 0.75 and random_num >= 0.5:
                mask_box = self.square(features, 4)
            else:
                mask_box = self.square(features, 5)
        else:
            raise Exception("Unknown mask shape name")
        return mask_box

    def hstripe(self, features, mask_size):
        """
        """
        # horizontal stripe
        mask_x1 = 0
        mask_x2 = features.shape[2]
        y1_max = features.shape[3] - mask_size
        mask_y1 = torch.randint(y1_max, (1,))
        mask_y2 = mask_y1 + mask_size
        new_idx = torch.randperm(features.shape[0])
        mask_box = (new_idx, mask_x1, mask_x2, mask_y1, mask_y2)
        return mask_box

    def vstripe(self, features, mask_size):
        """
        """
        # vertical stripe
        mask_y1 = 0
        mask_y2 = features.shape[3]
        x1_max = features.shape[2] - mask_size
        mask_x1 = torch.randint(x1_max, (1,))
        mask_x2 = mask_x1 + mask_size
        new_idx = torch.randperm(features.shape[0])
        mask_box = (new_idx, mask_x1, mask_x2, mask_y1, mask_y2)
        return mask_box

    def square(self, features, mask_size):
        """
        """
        # square
        x1_max = features.shape[2] - mask_size
        y1_max = features.shape[3] - mask_size
        mask_x1 = torch.randint(x1_max, (1,))
        mask_y1 = torch.randint(y1_max, (1,))
        mask_x2 = mask_x1 + mask_size
        mask_y2 = mask_y1 + mask_size
        new_idx = torch.randperm(features.shape[0])
        mask_box = (new_idx, mask_x1, mask_x2, mask_y1, mask_y2)
        return mask_box


class exchange_patch:
    def __init__(self, shape='stripe', mask_size=2, mode='random_direct'):
        self.shape = shape
        self.mask_size = mask_size
        self.mode = mode

    def __call__(self, features):
        # Stripe mask
        if self.shape == 'stripe':
            if self.mode == 'horizontal':
                features = self.xpatch_hstripe(features, self.mask_size)
            elif self.mode == 'vertical':
                features = self.xpatch_vstripe(features, self.mask_size)
            elif self.mode == 'random_direction':
                if random.random() < 0.5:
                    features = self.xpatch_hstripe(features, self.mask_size)
                else:
                    features = self.xpatch_vstripe(features, self.mask_size)
            else:
                raise Exception("Unknown stripe mask mode name")
        # Square mask
        elif self.shape == 'square':
            if self.mode == 'random_size':
                self.mask_size = 4 if random.random() < 0.5 else 5
            features = self.xpatch_square(features, self.mask_size)
        # Random stripe/square mask
        elif self.shape == 'random':
            random_num = random.random()
            if random_num < 0.25:
                features = self.xpatch_hstripe(features, 2)
            elif random_num < 0.5 and random_num >= 0.25:
                features = self.xpatch_vstripe(features, 2)
            elif random_num < 0.75 and random_num >= 0.5:
                features = self.xpatch_square(features, 4)
            else:
                features = self.xpatch_square(features, 5)
        else:
            raise Exception("Unknown mask shape name")

        return features

    def xpatch_hstripe(self, features, mask_size):
        """
        """
        # horizontal stripe
        y1_max = features.shape[3] - mask_size
        num_masks = 1
        for i in range(num_masks):
            mask_y1 = torch.randint(y1_max, (1,))
            mask_y2 = mask_y1 + mask_size
            new_idx = torch.randperm(features.shape[0])
            features[:, :, :, mask_y1 : mask_y2] = features[new_idx, :, :, mask_y1 : mask_y2]
        return features


    def xpatch_vstripe(self, features, mask_size):
        """
        """
        # vertical stripe
        x1_max = features.shape[2] - mask_size
        num_masks = 1
        for i in range(num_masks):
            mask_x1 = torch.randint(x1_max, (1,))
            mask_x2 = mask_x1 + mask_size
            new_idx = torch.randperm(features.shape[0])
            features[:, :, mask_x1 : mask_x2, :] = features[new_idx, :, mask_x1 : mask_x2, :]
        return features


    def xpatch_square(self, features, mask_size):
        """
        """
        # square
        x1_max = features.shape[2] - mask_size
        y1_max = features.shape[3] - mask_size
        num_masks = 1
        for i in range(num_masks):
            mask_x1 = torch.randint(x1_max, (1,))
            mask_y1 = torch.randint(y1_max, (1,))
            mask_x2 = mask_x1 + mask_size
            mask_y2 = mask_y1 + mask_size
            new_idx = torch.randperm(features.shape[0])
            features[:, :, mask_x1 : mask_x2, mask_y1 : mask_y2] = features[new_idx, :, mask_x1 : mask_x2, mask_y1 : mask_y2]
        return features


class cutout_patch:
    def __init__(self, mask_size=2):
        self.mask_size = mask_size

    def __call__(self, features):
        if random.random() < 0.5:
            y1_max = features.shape[3] - self.mask_size
            num_masks = 1
            for i in range(num_masks):
                mask_y1 = torch.randint(y1_max, (features.shape[0],))
                mask_y2 = mask_y1 + self.mask_size
                for k in range(features.shape[0]):
                    features[k, :, :, mask_y1[k] : mask_y2[k]] = 0
        else:
            x1_max = features.shape[3] - self.mask_size
            num_masks = 1
            for i in range(num_masks):
                mask_x1 = torch.randint(x1_max, (features.shape[0],))
                mask_x2 = mask_x1 + self.mask_size
                for k in range(features.shape[0]):
                    features[k, :, mask_x1[k] : mask_x2[k], :] = 0

        return features


class erase_patch:
    def __init__(self, mask_size=2):
        self.mask_size = mask_size

    def __call__(self, features):
        std, mean = torch.std_mean(features.detach())
        dim = features.shape[1]
        if random.random() < 0.5:
            y1_max = features.shape[3] - self.mask_size
            num_masks = 1
            for i in range(num_masks):
                mask_y1 = torch.randint(y1_max, (features.shape[0],))
                mask_y2 = mask_y1 + self.mask_size
                for k in range(features.shape[0]):
                    features[k, :, :, mask_y1[k] : mask_y2[k]] = torch.normal(mean.repeat(dim,14,2), std.repeat(dim,14,2))
        else:
            x1_max = features.shape[3] - self.mask_size
            num_masks = 1
            for i in range(num_masks):
                mask_x1 = torch.randint(x1_max, (features.shape[0],))
                mask_x2 = mask_x1 + self.mask_size
                for k in range(features.shape[0]):
                    features[k, :, mask_x1[k] : mask_x2[k], :] = torch.normal(mean.repeat(dim,2,14), std.repeat(dim,2,14))

        return features

class mixup_patch:
    def __init__(self, mask_size=2):
        self.mask_size = mask_size

    def __call__(self, features):
        lam = random.uniform(0, 1)
        if random.random() < 0.5:
            y1_max = features.shape[3] - self.mask_size
            num_masks = 1
            for i in range(num_masks):
                mask_y1 = torch.randint(y1_max, (1,))
                mask_y2 = mask_y1 + self.mask_size
                new_idx = torch.randperm(features.shape[0])
                features[:, :, :, mask_y1 : mask_y2] = lam*features[:, :, :, mask_y1 : mask_y2] + (1-lam)*features[new_idx, :, :, mask_y1 : mask_y2]
        else:
            x1_max = features.shape[2] - self.mask_size
            num_masks = 1
            for i in range(num_masks):
                mask_x1 = torch.randint(x1_max, (1,))
                mask_x2 = mask_x1 + self.mask_size
                new_idx = torch.randperm(features.shape[0])
                features[:, :, mask_x1 : mask_x2, :] = lam*features[:, :, mask_x1 : mask_x2, :] + (1-lam)*features[new_idx, :, mask_x1 : mask_x2, :]

        return features


class jigsaw_patch:
    def __init__(self, shift=5, group=2):
        self.shift = shift
        self.group = group

    def __call__(self, features):
        batchsize = features.size(0)
        dim = features.size(1)
        features = features.view(batchsize, dim, -1)

        # Shift Operation
        feature_random = torch.cat([features[:, :, self.shift:], features[:, :, :self.shift]], dim=2)
        x = feature_random

        # Patch Shuffle Operation
        try:
            x = x.view(batchsize, dim, self.group, -1)
        except:
            x = torch.cat([x, x[:, -2:-1, :]], dim=1)
            x = x.view(batchsize, self.group, -1, dim)

        x = torch.transpose(x, 2, 3).contiguous()

        x = x.view(batchsize, dim, -1)
        x = x.view(batchsize, dim, 16, 8)

        return x
from functools import partial

class TransformerHeadVit(nn.Module):
    def __init__(self,in_planes):
        super().__init__()
        self.blocks = nn.ModuleList([
            BlockVit(
                dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(2)])
        self.proj=nn.Conv2d(in_planes, 768, kernel_size=1, stride=1)
    def forward(self,x):
        B,C,H,W=x.shape
        x=self.proj(x).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x =checkpoint.checkpoint(blk,x)
        return x.transpose(1,2).reshape(B,-1,H,W)
        
class MlpVit(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionVit(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockVit(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionVit(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpVit(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
