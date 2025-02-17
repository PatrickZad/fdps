# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as tF
from torch import nn
import torch.utils.checkpoint as checkpoint
import math
from psd2.modeling.matcher import DDetrHungarianMatcher as Matcher
from .base import SearchBase
from psd2.structures.boxes import Boxes
from psd2.modeling.transformer import DeformableTransformer,DabDeformableTransformer
import copy
import cv2
import numpy as np
from ..build import META_ARCH_REGISTRY
from psd2.config import configurable
from psd2.modeling.position_encoding import PositionEmbeddingSine
from psd2.structures import NestedTensor
from psd2.utils.events import get_event_storage
from psd2.layers.set_criterion import DDetrSetCriterion as SetCriterion
from psd2.structures.boxes import box_cxcywh_to_xyxy
from psd2.layers.pooling import *
from psd2.modeling.reid_heads.id_assign import build_id_assigner
from psd2.modeling.reid_heads.box_augmentation import build_box_augmentor
from psd2.modeling.poolers import ROIPooler
from psd2.layers.mem_matching_losses import build_loss_layer
from psd2.structures import ImageList, Instances
from psd2.utils.visualizer import Visualizer, mlvl_pca_feat

@META_ARCH_REGISTRY.register()
class DDETR_Baseline(SearchBase):
    """This is the Deformable DETR module that performs person search"""

    @configurable
    def __init__(
        self,
        det_head,
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
        *args,**kws
    ):
        super().__init__(*args,**kws)    
        self.det_head=det_head    
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
        res = super().from_config(cfg)
        reid_cfg = cfg.REID_HEAD
        reid_pooler_cfg = reid_cfg.ROI_POOLER
        res["reid_pooler"] = ROIPooler(
            output_size=reid_pooler_cfg.POOLER_RESOLUTION,
            scales=reid_pooler_cfg.SCALES,
            sampling_ratio=reid_pooler_cfg.SAMP_RATIO,
            pooler_type=reid_pooler_cfg.TYPE,
            output_channels=1024,  #  compatible with deformRoiPool
        )
        loss_cfg = reid_cfg.LOSS
        loss_layer = build_loss_layer(loss_cfg, reid_cfg.PERSON_FEATURE.DIM)
        res["reid_loss"] = loss_layer
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
        elif pool_layer=="attn":
            pfeat_pooling=AttentionPool2d(reid_pooler_cfg.POOLER_RESOLUTION,embed_dim=2048,num_heads=2048//64,output_dim=reid_cfg.PERSON_FEATURE.DIM)
        else:
            raise KeyError(f"{pool_layer} is not supported!")
        res["pfeat_pooling"] = pfeat_pooling
        res["train_task"] = "ps" if reid_cfg.TRAIN_REID else "det"
        res["pid_assigner"] = build_id_assigner(reid_cfg.ID_ASSIGN)
        res["cws"] = cfg.CWS
        res["train_with_nms"] = reid_cfg.TRAIN_WITH_NMS
        res["train_with_det"] = reid_cfg.TRAIN_WITH_DET
        res["box_augmentor"] = (
            build_box_augmentor(reid_cfg.BOX_AUGMENTATION)
            if reid_cfg.BOX_AUGMENTATION.ENABLE
            else None
        )
        res["use_checkpoint"] = reid_cfg.USE_CHECKPOINT
        res["det_head"]=DDetrDetHead(cfg,res["backbone"].output_shape())
        return res


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
            org_pids[org_pids == -1] = 0  # unlabeled people pid =0
            inst_img.gt_classes = org_pids.long()
            inst_img.gt_classes = org_pids.long()
            inst_img._org_hw = (input_dict["org_height"], input_dict["org_width"])
            inst_img.org_boxes = Boxes(torch.tensor(input_dict["org_boxes"]))
            gt_instances.append(inst_img.to(self.device))
        return (
            ImageList.from_tensors(images, self.backbone.size_divisibility),
            gt_instances,
        )

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
            # print("det backbone time: {}".format(time.time()-t0))
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
            det_pred_instances,det_losses= self.det_head(
                image_list, features,gt_instances
            )
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
                        gt_id[gt_id == 0] = -1
                        gt_id[gt_id > 0] -= 2
                    self.visualize_training(
                        (image_list, gt_instances),
                        features[list(features.keys())[-1]],
                        outputs,
                    )
            losses = {}
            losses.update(det_losses)
            return losses
        elif self.training and self.train_task == "ps":
            if not self.train_with_det:
                return {
                    "pred_boxes": [],
                    "pred_scores": [],
                    "pred_pids": [],
                }
            raise NotImplementedError
        else:
            det_pred_instances, _ = self.det_head(
                image_list, features, None
            )
            # print("det pred time: {}".format(time.time()-t0))
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
        # print("reid backbone time: {}".format(time.time()-t0))
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
                    if isinstance(pos_featmaps,torch.Tensor):
                        vis_feat=pos_featmaps
                    else:
                        vis_feat=pos_featmaps[-1]
                    self.visualize_training_ps(
                        image_list.tensor,
                        vis_feat.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
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
            # print("reid pred time: {}".format(time.time()-t0))
            return p_embs
    def get_reid_backbone_features(self, det_backbone_features, image_list):
        raise NotImplementedError

    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        raise NotImplementedError
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

        threds = [0.1]
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
        if isinstance(featmap, torch.Tensor):
            featmap = featmap[None]
        if isinstance(featmap, dict):
            featmap = list(featmap.values())
        level_pcas = []
        for featmap_i in featmap:
            level_pcas.append(mlvl_pca_feat(featmap_i[None])[0])
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

@META_ARCH_REGISTRY.register()
class DDETR_Baseline_JointDc(DDETR_Baseline):
    def get_ps_pos_samples(self,gts):
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
    def forward_det(self, image_list, features, gt_instances):
        if self.training:
            det_pred_instances,det_losses= self.det_head(
                image_list, features,gt_instances
            )
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
                        gt_id[gt_id == 0] = -1
                        gt_id[gt_id > 0] -= 2
                    self.visualize_training(
                        (image_list, gt_instances),
                        features[list(features.keys())[-1]],
                        outputs,
                    )
            losses = {}
            losses.update(det_losses)
            return losses
        else:
            det_pred_instances, _ = self.det_head(
                image_list, features, None
            )
            # print("det pred time: {}".format(time.time()-t0))
            det_outputs = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pi, pred_img in enumerate(det_pred_instances):
                boxes = pred_img.pred_boxes.tensor
                scores = pred_img.scores
                det_outputs["pred_boxes"].append(boxes)
                det_outputs["pred_scores"].append(scores.unsqueeze(1))
                det_outputs["reid_feats"].append(boxes.clone())  # dumm impl
            return det_outputs
    def forward_ps(self, det_backbone_features, image_list, gts,det_pred=None):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        # print("reid backbone time: {}".format(time.time()-t0))
        if self.training:
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples( gts)

            pos_featmaps = self.get_reid_person_features(
                reid_bk_feats, pos_boxes
            )  # nf x c x h x w

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    if isinstance(pos_featmaps,torch.Tensor):
                        vis_feat=pos_featmaps
                    else:
                        vis_feat=pos_featmaps[-1]
                    self.visualize_training_ps(
                        image_list.tensor,
                        vis_feat.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
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
            # print("reid pred time: {}".format(time.time()-t0))
            return p_embs
    
    def forward(self, input_list):
        if self.training:
            images, gt_instances = self.preprocess_input(input_list)
            det_bk_features = self.backbone(images.tensor)
            losses= self.forward_det(images, det_bk_features, gt_instances)  # losses
            # back to original pid
            for gti in gt_instances:
                cur_pids = gti.gt_classes
                gti.gt_classes[cur_pids == 0] = -1
                gti.gt_classes[cur_pids > 0] -= 2
            ps_losses= self.forward_ps(det_bk_features, images, gt_instances)
            losses.update(ps_losses)
            return losses
        else:
            if "query" in input_list[0]:
                return self.inf_query(input_list)
            images, gt_instances = self.preprocess_input(input_list)
            det_bk_features = self.backbone(images.tensor)
            # print("det backbone time: {}".format(time.time()-t0))
            det_pred = self.forward_det(images, det_bk_features, gt_instances)
            reid_feats = self.forward_ps(
                det_bk_features, images, gt_instances, det_pred=det_pred
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

@META_ARCH_REGISTRY.register()
class DnDabDDETR_Baseline(DDETR_Baseline):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["det_head"]=DnDabDDetrDetHead(cfg,ret["backbone"].output_shape())
        return ret

@META_ARCH_REGISTRY.register()
class DabDDETR_Baseline(DDETR_Baseline):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["det_head"]=DabDDetrDetHead(cfg,ret["backbone"].output_shape())
        return ret

@META_ARCH_REGISTRY.register()
class DDDETR_Baseline(DDETR_Baseline):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["det_head"]=DDDetrDetHead(cfg,ret["backbone"].output_shape())
        return ret

@META_ARCH_REGISTRY.register()
class DDETR_NextBaseline(DDETR_Baseline):
    pass

from psd2.modeling.backbone.resnet import (
    ResNet,
    BottleneckBlock,
)
import torch.nn.init as init
import copy
@META_ARCH_REGISTRY.register()
class DDETR_C4Side(DDETR_Baseline):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super().__init__(**kwargs)

        self.side_init = side_init
        self.alpha_res3 = nn.Parameter(torch.tensor(0.0))
        self.alpha_res4 = nn.Parameter(torch.tensor(0.0))
        # side reid network
        self.bn_neck = bn_neck
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
        self.side_res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "BN",
                    "bottleneck_channels": 512,
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
        for p in self.det_head.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg):
        res = super().from_config(cfg)
        bn_neck = nn.BatchNorm1d(cfg.REID_HEAD.PERSON_FEATURE.DIM)
        init.constant_(bn_neck.bias, 0.0)
        init.normal_(bn_neck.weight, std=0.01)
        bn_neck.bias.requires_grad_(False)
        res["bn_neck"] = bn_neck
        res["side_init"] = cfg.REID_HEAD.INIT_WEIGHT
        return res

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
            self.side_res5.load_state_dict(side_res_params[2])
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

    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats = self.reid_pooler([reid_backbone_features], d2_boxes)
        roi_feats = (
            checkpoint.checkpoint(self.side_res5, roi_feats)
            if self.use_checkpoint and self.training
            else self.side_res5(roi_feats)
        )  # n x c x h x w
        return roi_feats

    def get_reid_embed(self, p_feat_maps):
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        return tF.normalize(pfeat_embs, dim=-1)

@META_ARCH_REGISTRY.register()
class DnDabDDETR_C4Side(DDETR_C4Side):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["det_head"]=DnDabDDetrDetHead(cfg,ret["backbone"].output_shape())
        return ret
@META_ARCH_REGISTRY.register()
class DabDDETR_C4Side(DDETR_C4Side):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["det_head"]=DabDDetrDetHead(cfg,ret["backbone"].output_shape())
        return ret
@META_ARCH_REGISTRY.register()
class DDETR_C4SideMmean(DDETR_C4Side):
    def forward_ps(self, det_pred, det_backbone_features, image_list, gts):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        # print("reid backbone time: {}".format(time.time()-t0))
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
                    if isinstance(pos_featmaps,torch.Tensor):
                        vis_feat=pos_featmaps
                    else:
                        vis_feat=pos_featmaps[-1]
                    self.visualize_training_ps(
                        image_list.tensor,
                        vis_feat.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(pos_featmaps.split([bxs.shape[0] for bxs in pos_boxes]))
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            return self.reid_loss(pos_embs, pos_ids, None)
        else:
            num_boxes_per_image = [bi.shape[0] for bi in det_pred["pred_boxes"]]
            p_featmaps = self.get_reid_person_features(
                reid_bk_feats, det_pred["pred_boxes"]
            )
            del reid_bk_feats
            p_embs = self.get_reid_embed(p_featmaps.split(num_boxes_per_image))
            del p_featmaps
            if self.cws:
                cat_scores = torch.cat(det_pred["pred_scores"], dim=0)
                p_embs *= cat_scores  # .unsqueeze(1)
            p_embs = torch.split(p_embs, num_boxes_per_image)
            # print("reid pred time: {}".format(time.time()-t0))
            return p_embs
    def get_reid_embed(self, p_feat_maps):
        n_bs=[feats.shape[0] for feats in p_feat_maps]
        p_feat_maps=torch.cat(p_feat_maps)
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        pfeat_embs=torch.split(pfeat_embs,n_bs)
        out_pfeat_embs=[]
        for embs in pfeat_embs:
            out_pfeat_embs.append(embs-torch.mean(embs,dim=0,keepdim=True))
        pfeat_embs=torch.cat(out_pfeat_embs)
        return tF.normalize(pfeat_embs, dim=-1)
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = tF.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DDetrDetHead(nn.Module):
    @configurable
    def __init__(self,in_features,input_shape,pos_encoding,num_queries,num_feature_levels,transformer,use_aux_loss,with_box_refine,criterion,use_checkpoint ) -> None:
        super().__init__()
        self.in_features=in_features
        self.num_feature_levels=num_feature_levels
        self.pos_encoding = pos_encoding
        self.transformer=transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.class_embed = nn.Linear(hidden_dim, 1)  # only 1 category
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if self.num_feature_levels > 1:
            num_backbone_outs = len(in_features)
            input_proj_list = []
            for _ in in_features:
                in_channels = input_shape[_].channels
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            in_channels = input_shape[in_features[-1]].channels
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.aux_loss = use_aux_loss
        self.with_box_refine = with_box_refine
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(1) * bias_value  # only 1 category
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        
        num_pred = self.transformer.decoder.num_layers
        
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        
        self.criterion = criterion
        self.num_queries = num_queries
        self.use_checkpoint=use_checkpoint
    @classmethod
    def from_config(cls, cfg,input_shape):
        det_cfg = cfg.DETECTOR
        detr_cfg = det_cfg.MODEL
        trans_cfg = detr_cfg.D_TRANSFORMER
        N_steps = trans_cfg.HIDDEN_DIM // 2
        pos_enc = PositionEmbeddingSine(N_steps, normalize=True)
        transformer = DeformableTransformer(
            d_model=trans_cfg.HIDDEN_DIM,
            nhead=trans_cfg.N_HEADS,
            num_encoder_layers=trans_cfg.ENC_DEPTH,
            num_decoder_layers=trans_cfg.DEC_DEPTH,
            dim_feedforward=trans_cfg.DIM_FEEDFORWARD,
            dropout=trans_cfg.DROPOUT,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=detr_cfg.NUM_FEAT_LEVELS,
            dec_n_points=trans_cfg.DEC_N_POINTS,
            enc_n_points=trans_cfg.ENC_N_POINTS,
            two_stage=False,
            two_stage_num_proposals=detr_cfg.N_QUERIES,
        )
        # Loss parameters:
        loss_cfg = det_cfg.LOSS
        loss_weights = loss_cfg.LOSS_WEIGHTS
        class_weight = loss_weights.CLS
        giou_weight = loss_weights.BOX_GIOU
        l1_weight = loss_weights.BOX_L1
        no_object_weight = loss_weights.NO_OBJECT
        deep_supervision = loss_cfg.DEEP_SUPERVISION
        use_focal = loss_cfg.FOCAL.USE_FOCAL
        focal_alpha = loss_cfg.FOCAL.ALPHA
        focal_gamma = loss_cfg.FOCAL.GAMMA
        matcher = Matcher(
            cost_class=class_weight,
            cost_bbox=l1_weight,
            cost_giou=giou_weight,
            use_focal=use_focal,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        weight_dict = {
            "loss_ce": class_weight,
            "loss_bbox": l1_weight,
            "loss_giou": giou_weight,
        }
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(trans_cfg.DEC_DEPTH - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f"_enc": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes"]
        criterion = SetCriterion(
            num_classes=det_cfg.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            use_focal=use_focal,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        return {
            "input_shape": input_shape,
            "pos_encoding": pos_enc,
            "transformer": transformer,
            "in_features": detr_cfg.IN_FEATS,
            "num_queries": detr_cfg.N_QUERIES,
            "use_aux_loss": deep_supervision,
            "with_box_refine": detr_cfg.BOX_REFINE,
            "criterion": criterion,
            "num_feature_levels": detr_cfg.NUM_FEAT_LEVELS,
            "use_checkpoint": detr_cfg.USE_CHECKPOINT
        }
    def forward(self,image_list,bk_features,gt_instances=None):
        features = []
        pos = []
        for name in self.in_features:
            x = bk_features[name]
            masks = tF.interpolate(image_list.mask.float()[None], size=(x.shape[-2],x.shape[-1])).to(torch.bool)[0]
            nested_feat = NestedTensor(x, masks)
            features.append(nested_feat)
            pos.append(self.pos_encoding(nested_feat))
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                mask = tF.interpolate(image_list.mask.float()[None], size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.pos_encoding(NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.weight
        (
            aux_info,
            hs,
            init_reference,
            inter_references,
            memory,
        ) = self.transformer(srcs, masks, pos, query_embeds,use_checkpoint=self.use_checkpoint)
        del aux_info
        del memory

        outputs_classes = []
        outputs_coords = []
        outputs_rpts = []
        outputs_embs = []
        inv_mask = 1 - image_list.mask.int()
        img_sizes=torch.stack(
            [inv_mask.sum(dim=1)[:, 0], inv_mask.sum(dim=2)[:, 0]], dim=-1
        ).tolist()  # B x 2 hw
        img_whwh = image_list.mask.new_tensor(
            [hw[::-1] * 2 for hw in img_sizes],dtype=torch.float
        )
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = _inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
                outputs_rpts.append(
                    reference[..., :2].sigmoid() * img_whwh[:, None, :2]
                )  # rel in aug (before padding)->abs in aug
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                outputs_rpts.append(reference.sigmoid() * img_whwh[:, None, :2])
            outputs_coord = tmp.sigmoid()  # B x N x 4
            outputs_coord = outputs_coord * img_whwh[:, None, :]  # ccwh_abs
            bimg, nq = outputs_coord.shape[:2]
            outputs_coord = box_cxcywh_to_xyxy(outputs_coord.flatten(0, 1)).reshape(
                bimg, nq, -1
            )
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)  # xyxy_abs
            outputs_embs.append(hs[lvl])

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_rpts = torch.stack(outputs_rpts)
        outputs_embs = torch.stack(outputs_embs)

        all_output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_embs": outputs_embs[-1],
            "pred_ref_pts": outputs_rpts[-1],  # for vis only
        }  # xyxy boxes in aug
        all_output["aux_outputs"] = [
            {"pred_logits": a, "pred_boxes": b, "pred_embs": c, "pred_ref_pts": d}
            for a, b, c, d in zip(
                outputs_class[:-1],
                outputs_coord[:-1],
                outputs_embs[:-1],
                outputs_rpts[:-1],
            )
        ]
        with torch.no_grad():
            pred_instances=[]
            for i,isize in enumerate(image_list.image_sizes):
                result = Instances(isize)
                result.pred_boxes = Boxes(outputs_coord[-1][i])
                result.scores = outputs_class[-1][i].sigmoid()[...,0]
                result.pred_classes = torch.zeros_like(result.scores,dtype=torch.long)
                pred_instances.append(result)
        if self.training:
            targets = []
            for inst in gt_instances:
                targets.append(
                    {
                        "file_name": inst._file_name,
                        "image_id": inst._image_id,
                        "boxes": inst.gt_boxes.tensor,
                        "labels": torch.zeros(inst.gt_boxes.tensor.shape[0], dtype=torch.long).to(
                            outputs_class.device
                        ),
                        "aug_whwh": inst.gt_boxes.tensor.new_tensor(((inst._image_size[1], inst._image_size[0]) * 2,)),
                    }
                )
            # det losses
            _, loss_dict = self.criterion(all_output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            
            return pred_instances,loss_dict
        else:
            return pred_instances,{}
# from psd2.layers.deform_conv import DeformConvPack
class DDDetrDetHead(DDetrDetHead):
    @configurable
    def __init__(self,*args,**kws ) -> None:
        super().__init__(*args,**kws)
        hidden_dim = self.transformer.d_model
        in_features=kws["in_features"]
        input_shape=kws["input_shape"]
        if self.num_feature_levels > 1:
            num_backbone_outs = len(in_features)
            input_proj_list = []
            for _ in in_features:
                in_channels = input_shape[_].channels
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        DeformConvPack(hidden_dim,hidden_dim,3,padding=1,
                        stride=1,)
                    )
                )
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        DeformConvPack(hidden_dim,hidden_dim,3,padding=1,
                        stride=2,)
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            in_channels = input_shape[in_features[-1]].channels
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        DeformConvPack(hidden_dim,hidden_dim,3,padding=1,
                        stride=1,)
                    )
                ]
            )


class DabDDetrDetHead(DDetrDetHead):
    @configurable
    def __init__(self,*args,**kws ) -> None:
        super().__init__(*args,**kws)
        hidden_dim = self.transformer.d_model
        self.tgt_embed = nn.Embedding(self.num_queries, hidden_dim) 
        self.refpoint_embed = nn.Embedding(self.num_queries, 4)

        del self.query_embed
    @classmethod
    def from_config(cls, cfg,input_shape):
        ret=DDetrDetHead.from_config(cfg,input_shape)
        det_cfg = cfg.DETECTOR
        detr_cfg = det_cfg.MODEL
        trans_cfg = detr_cfg.D_TRANSFORMER
        transformer = DabDeformableTransformer(
            d_model=trans_cfg.HIDDEN_DIM,
            nhead=trans_cfg.N_HEADS,
            num_encoder_layers=trans_cfg.ENC_DEPTH,
            num_decoder_layers=trans_cfg.DEC_DEPTH,
            dim_feedforward=trans_cfg.DIM_FEEDFORWARD,
            dropout=trans_cfg.DROPOUT,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=detr_cfg.NUM_FEAT_LEVELS,
            dec_n_points=trans_cfg.DEC_N_POINTS,
            enc_n_points=trans_cfg.ENC_N_POINTS,
        )
        ret["transformer"]=transformer
        # Loss parameters:
        loss_cfg = det_cfg.LOSS
        loss_weights = loss_cfg.LOSS_WEIGHTS
        class_weight = loss_weights.CLS
        giou_weight = loss_weights.BOX_GIOU
        l1_weight = loss_weights.BOX_L1
        no_object_weight = loss_weights.NO_OBJECT
        deep_supervision = loss_cfg.DEEP_SUPERVISION
        use_focal = loss_cfg.FOCAL.USE_FOCAL
        focal_alpha = loss_cfg.FOCAL.ALPHA
        focal_gamma = loss_cfg.FOCAL.GAMMA
        matcher = Matcher(
            cost_class=class_weight,
            cost_bbox=l1_weight,
            cost_giou=giou_weight,
            use_focal=use_focal,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        weight_dict = {
            "loss_ce": class_weight/2, # refer to the released config, 2.0 for matching, 1.0 for learning
            "loss_bbox": l1_weight,
            "loss_giou": giou_weight,
        }
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(trans_cfg.DEC_DEPTH - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f"_enc": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes"]
        criterion = SetCriterion(
            num_classes=det_cfg.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            use_focal=use_focal,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        ret["criterion"]=criterion
        return ret
    
    def forward(self,image_list,bk_features,gt_instances=None):
        features = []
        pos = []
        for name in self.in_features:
            x = bk_features[name]
            masks = tF.interpolate(image_list.mask.float()[None], size=(x.shape[-2],x.shape[-1])).to(torch.bool)[0]
            nested_feat = NestedTensor(x, masks)
            features.append(nested_feat)
            pos.append(self.pos_encoding(nested_feat))
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    if self.use_checkpoint and not self.training:
                        src=checkpoint.checkpoint(self.input_proj[l],features[-1].tensors)
                    else:
                        src = self.input_proj[l](features[-1].tensors)
                else:
                    if self.use_checkpoint and not self.training:
                        src=checkpoint.checkpoint(self.input_proj[l],srcs[-1])
                    else:
                        src = self.input_proj[l](srcs[-1])
                mask = tF.interpolate(image_list.mask.float()[None], size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.pos_encoding(NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        tgt_embed = self.tgt_embed.weight           # nq, 256
        refanchor = self.refpoint_embed.weight      # nq, 4
        query_embeds = torch.cat((tgt_embed, refanchor), dim=1)

        (
            aux_info,
            hs,
            init_reference,
            inter_references,
            memory,
        ) = self.transformer(srcs, masks, pos, query_embeds,use_checkpoint=self.use_checkpoint)
        del aux_info
        del memory

        outputs_classes = []
        outputs_coords = []
        inv_mask = 1 - image_list.mask.int()
        img_sizes=torch.stack(
            [inv_mask.sum(dim=1)[:, 0], inv_mask.sum(dim=2)[:, 0]], dim=-1
        ).tolist()  # B x 2 hw
        img_whwh = image_list.mask.new_tensor(
            [hw[::-1] * 2 for hw in img_sizes],dtype=torch.float
        )
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = _inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()  # B x N x 4
            outputs_coord = outputs_coord * img_whwh[:, None, :]  # ccwh_abs
            bimg, nq = outputs_coord.shape[:2]
            outputs_coord = box_cxcywh_to_xyxy(outputs_coord.flatten(0, 1)).reshape(
                bimg, nq, -1
            )
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)  # xyxy_abs

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        all_output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }  # xyxy boxes in aug
        all_output["aux_outputs"] = [
            {"pred_logits": a, "pred_boxes": b,}
            for a, b in zip(
                outputs_class[:-1],
                outputs_coord[:-1],
            )
        ]
        with torch.no_grad():
            pred_instances=[]
            for i,isize in enumerate(image_list.image_sizes):
                result = Instances(isize)
                result.pred_boxes = Boxes(outputs_coord[-1][i])
                result.scores = outputs_class[-1][i].sigmoid()[...,0]
                result.pred_classes = torch.zeros_like(result.scores,dtype=torch.long)
                pred_instances.append(result)
        if self.training:
            # det losses
            targets = []
            for inst in gt_instances:
                targets.append(
                    {
                        "file_name": inst._file_name,
                        "image_id": inst._image_id,
                        "boxes": inst.gt_boxes.tensor,
                        "labels": torch.zeros(inst.gt_boxes.tensor.shape[0], dtype=torch.long).to(
                            self.tgt_embed.weight.device
                        ),
                        "aug_whwh": inst.gt_boxes.tensor.new_tensor(((inst._image_size[1], inst._image_size[0]) * 2,)),
                    }
                )
            _, loss_dict = self.criterion(all_output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            
            return pred_instances,loss_dict
        else:
            return pred_instances,{}



class DnDabDDetrDetHead(DDetrDetHead):
    @configurable
    def __init__(self,dn_scalar,label_noise_scale,box_noise_scale,*args,**kws ) -> None:
        super().__init__(*args,**kws)
        num_classes=1
        hidden_dim = self.transformer.d_model
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
        self.tgt_embed = nn.Embedding(self.num_queries, hidden_dim-1)  # for indicator
        self.refpoint_embed = nn.Embedding(self.num_queries, 4)

        del self.query_embed

        # dn args
        self.dn_scalar=dn_scalar
        self.label_noise_scale=label_noise_scale
        self.box_noise_scale=box_noise_scale


    @classmethod
    def from_config(cls, cfg,input_shape):
        ret=DDetrDetHead.from_config(cfg,input_shape)
        det_cfg = cfg.DETECTOR
        detr_cfg = det_cfg.MODEL
        trans_cfg = detr_cfg.D_TRANSFORMER
        transformer = DabDeformableTransformer(
            d_model=trans_cfg.HIDDEN_DIM,
            nhead=trans_cfg.N_HEADS,
            num_encoder_layers=trans_cfg.ENC_DEPTH,
            num_decoder_layers=trans_cfg.DEC_DEPTH,
            dim_feedforward=trans_cfg.DIM_FEEDFORWARD,
            dropout=trans_cfg.DROPOUT,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=detr_cfg.NUM_FEAT_LEVELS,
            dec_n_points=trans_cfg.DEC_N_POINTS,
            enc_n_points=trans_cfg.ENC_N_POINTS,
        )
        ret["transformer"]=transformer
        # Loss parameters:
        loss_cfg = det_cfg.LOSS
        loss_weights = loss_cfg.LOSS_WEIGHTS
        class_weight = loss_weights.CLS
        giou_weight = loss_weights.BOX_GIOU
        l1_weight = loss_weights.BOX_L1
        no_object_weight = loss_weights.NO_OBJECT
        deep_supervision = loss_cfg.DEEP_SUPERVISION
        use_focal = loss_cfg.FOCAL.USE_FOCAL
        focal_alpha = loss_cfg.FOCAL.ALPHA
        focal_gamma = loss_cfg.FOCAL.GAMMA
        matcher = Matcher(
            cost_class=class_weight,
            cost_bbox=l1_weight,
            cost_giou=giou_weight,
            use_focal=use_focal,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        weight_dict = {
            "loss_ce": class_weight/2, # refer to the released config, 2.0 for matching, 1.0 for learning,
            "loss_bbox": l1_weight,
            "loss_giou": giou_weight,
            "loss_ce_dn": class_weight/2, # refer to the released config, 2.0 for matching, 1.0 for learning,
            "loss_bbox_dn": l1_weight,
            "loss_giou_dn": giou_weight,
        }
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(trans_cfg.DEC_DEPTH - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f"_enc": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes"]
        criterion = SetCriterion(
            num_classes=det_cfg.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            use_focal=use_focal,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        ret["criterion"]=criterion
        dn_cfg=det_cfg.DN_DETR
        ret.update({"dn_scalar":dn_cfg.SCALAR,"label_noise_scale":dn_cfg.LABEL_NOISE_SCALE,"box_noise_scale":dn_cfg.BOX_NOISE_SCALE})
        return ret

    def prepare_for_dn(self, tgt_weight, embedweight, batch_size, targets ):
        """
        The major difference from DN-DAB-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, scalar, label_noise_scale, box_noise_scale, num_patterns
        :param tgt_weight: use learnbal tgt in dab deformable detr
        :param embedweight: positional anchor queries
        :param batch_size: bs
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """

        num_patterns = 1
        num_classes=1
        hidden_dim = self.transformer.d_model

        indicator0 = torch.zeros([self.num_queries * num_patterns, 1]).cuda()
        # sometimes the target is empty, add a zero part of label_enc to avoid unused parameters
        tgt = torch.cat([tgt_weight, indicator0], dim=1) + self.label_enc.weight[0][0]*torch.tensor(0).cuda()
        refpoint_emb = embedweight
        if self.training:
            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]
            # you can uncomment this to use fix number of dn queries
            # if int(max(known_num))>0:
            #     scalar=scalar//int(max(known_num))

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # add noise
            known_indice = known_indice.repeat(self.dn_scalar, 1).view(-1)
            known_labels = labels.repeat(self.dn_scalar, 1).view(-1)
            known_bid = batch_idx.repeat(self.dn_scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.dn_scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            # noise on the label
            if self.label_noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (self.label_noise_scale)).view(-1)  # usually half of bbox noise
                new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            # noise on the box
            if self.box_noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                            diff).cuda() * self.box_noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            m = known_labels_expaned.long().to('cuda')
            input_label_embed = self.label_enc(m)
            # add dn part indicator
            indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda()
            input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)
            input_bbox_embed = _inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.dn_scalar)
            padding_label = torch.zeros(pad_size, hidden_dim).cuda()
            padding_bbox = torch.zeros(pad_size, 4).cuda()
            input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
            input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)

            # map in order
            map_known_indice = torch.tensor([]).to('cuda')
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.dn_scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

            tgt_size = pad_size + self.num_queries * num_patterns
            attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.dn_scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.dn_scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
        else:  # no dn for inference
            input_query_label = tgt.repeat(batch_size, 1, 1)
            input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        # input_query_label = input_query_label.transpose(0, 1)
        # input_query_bbox = input_query_bbox.transpose(0, 1)

        return input_query_label, input_query_bbox, attn_mask, mask_dict
    
    def compute_dn_loss(self,mask_dict,aug_whwh_bs, aux_num):
        """
        compute dn loss in criterion
        Args:
            mask_dict: a dict for dn information
            training: training or inference flag
            aux_num: aux loss number
            focal_alpha:  for focal loss
        """
        losses = {}
        if 'output_known_lbs_bboxes' in mask_dict:
            output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
            known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
            map_known_indice = mask_dict['map_known_indice']

            known_indice = mask_dict['known_indice']

            batch_idx = mask_dict['batch_idx']
            bid = batch_idx[known_indice]
            if len(output_known_class) > 0:
                output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
                output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
                aug_whwh_bs=aug_whwh_bs[bid]
            num_tgt = known_indice.numel()
            losses.update(dn_tgt_loss_labels(output_known_class[-1], known_labels, num_tgt, self.criterion.focal_loss_alpha,self.criterion.focal_loss_gamma))
            losses.update(dn_tgt_loss_boxes(output_known_coord[-1], known_bboxs, aug_whwh_bs, num_tgt))
        else:
            losses['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            losses['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            losses['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')

        if aux_num:
            for i in range(aux_num):
                # dn aux loss
                if 'output_known_lbs_bboxes' in mask_dict:
                    l_dict = dn_tgt_loss_labels(output_known_class[i], known_labels, num_tgt, self.criterion.focal_loss_alpha,self.criterion.focal_loss_gamma)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    l_dict = dn_tgt_loss_boxes(output_known_coord[i], known_bboxs, aug_whwh_bs,num_tgt)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
    
    def forward(self,image_list,bk_features,gt_instances=None):
        features = []
        pos = []
        for name in self.in_features:
            x = bk_features[name]
            masks = tF.interpolate(image_list.mask.float()[None], size=(x.shape[-2],x.shape[-1])).to(torch.bool)[0]
            nested_feat = NestedTensor(x, masks)
            features.append(nested_feat)
            pos.append(self.pos_encoding(nested_feat))
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                mask = tF.interpolate(image_list.mask.float()[None], size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.pos_encoding(NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        tgt_all_embed = tgt_embed = self.tgt_embed.weight           # nq, 256
        refanchor = self.refpoint_embed.weight      # nq, 4

        if self.training:
            targets = []
            for inst in gt_instances:
                targets.append(
                    {
                        "file_name": inst._file_name,
                        "image_id": inst._image_id,
                        "boxes": inst.gt_boxes.tensor,
                        "labels": torch.zeros(inst.gt_boxes.tensor.shape[0], dtype=torch.long).to(
                            self.tgt_embed.weight.device
                        ),
                        "aug_whwh": inst.gt_boxes.tensor.new_tensor(((inst._image_size[1], inst._image_size[0]) * 2,)),
                    }
                )
        else:
            targets=None
        # prepare for dn
        input_query_label, input_query_bbox, attn_mask, mask_dict = \
            self.prepare_for_dn( tgt_all_embed, refanchor, src.size(0), targets)
        query_embeds = torch.cat((input_query_label, input_query_bbox), dim=2)

        (
            aux_info,
            hs,
            init_reference,
            inter_references,
            memory,
        ) = self.transformer(srcs, masks, pos, query_embeds,attn_mask,use_checkpoint=self.use_checkpoint)
        del aux_info
        del memory

        outputs_classes = []
        outputs_coords = []
        inv_mask = 1 - image_list.mask.int()
        img_sizes=torch.stack(
            [inv_mask.sum(dim=1)[:, 0], inv_mask.sum(dim=2)[:, 0]], dim=-1
        ).tolist()  # B x 2 hw
        img_whwh = image_list.mask.new_tensor(
            [hw[::-1] * 2 for hw in img_sizes],dtype=torch.float
        )
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = _inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()  # B x N x 4
            outputs_coord = outputs_coord * img_whwh[:, None, :]  # ccwh_abs
            bimg, nq = outputs_coord.shape[:2]
            outputs_coord = box_cxcywh_to_xyxy(outputs_coord.flatten(0, 1)).reshape(
                bimg, nq, -1
            )
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)  # xyxy_abs

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)


        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
            output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
            outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
            outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes']=(output_known_class,output_known_coord)

        all_output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }  # xyxy boxes in aug
        all_output["aux_outputs"] = [
            {"pred_logits": a, "pred_boxes": b,}
            for a, b in zip(
                outputs_class[:-1],
                outputs_coord[:-1],
            )
        ]
        with torch.no_grad():
            pred_instances=[]
            for i,isize in enumerate(image_list.image_sizes):
                result = Instances(isize)
                result.pred_boxes = Boxes(outputs_coord[-1][i])
                result.scores = outputs_class[-1][i].sigmoid()[...,0]
                result.pred_classes = torch.zeros_like(result.scores,dtype=torch.long)
                pred_instances.append(result)
        if self.training:
            # det losses
            _, loss_dict = self.criterion(all_output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            # dn loss
            aux_num = 0
            if 'aux_outputs' in all_output:
                aux_num = len(all_output['aux_outputs'])
            aug_whwh_bs = torch.cat([v["aug_whwh"] for v in targets], dim=0)
            dn_losses=self.compute_dn_loss(mask_dict,aug_whwh_bs,aux_num)
            weight_dict = self.criterion.weight_dict
            for k in dn_losses.keys():
                if k in weight_dict:
                    dn_losses[k] *= weight_dict[k]
            loss_dict.update(dn_losses)
            
            return pred_instances,loss_dict
        else:
            return pred_instances,{}


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
    attn_reshaped = (attn_reshaped**2).sum(0)
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
def prepare_for_dn_loss(mask_dict):
    """
    prepare dn components to calculate loss
    Args:
        mask_dict: a dict that contains dn information
    Returns:

    """
    output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
    known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
    map_known_indice = mask_dict['map_known_indice']

    known_indice = mask_dict['known_indice']

    batch_idx = mask_dict['batch_idx']
    bid = batch_idx[known_indice]
    if len(output_known_class) > 0:
        output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
    num_tgt = known_indice.numel()
    return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt
from psd2.layers.set_criterion import sigmoid_focal_loss_jit
import psd2.structures.boxes as iou_tools
def dn_tgt_loss_boxes(src_boxes, tgt_boxes,aug_whwh , num_tgt,):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    if len(tgt_boxes) == 0:
        return {
            'loss_bbox_dn': torch.as_tensor(0.).to('cuda'),
            'loss_giou_dn': torch.as_tensor(0.).to('cuda'),
        }
    loss_giou = 1 - torch.diag(iou_tools.generalized_box_iou(src_boxes,tgt_boxes))

    src_boxes_ = src_boxes / aug_whwh  # xyxy_rel
    target_boxes_ = tgt_boxes / aug_whwh  # xyxy_rel
    src_boxes_ = iou_tools.box_xyxy_to_cxcywh(src_boxes_)  # ccwh_rel
    target_boxes_ = iou_tools.box_xyxy_to_cxcywh(target_boxes_)  # ccwh_rel
    loss_bbox = tF.l1_loss(src_boxes_, target_boxes_, reduction="none")

    losses = {}
    losses['loss_bbox_dn'] = loss_bbox.sum() / num_tgt
    losses['loss_giou_dn'] = loss_giou.sum() / num_tgt
    return losses


def dn_tgt_loss_labels(src_logits_, tgt_labels_, num_tgt, focal_alpha,focal_gamma, ):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    if len(tgt_labels_) == 0:
        return {
            'loss_ce_dn': torch.as_tensor(0.).to('cuda'),
        }

    src_logits, tgt_labels= src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0)

    target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                        dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)

    target_classes_onehot = target_classes_onehot[:, :, :-1]
    loss_ce = sigmoid_focal_loss_jit(src_logits,target_classes_onehot,focal_alpha,focal_gamma,reduction="sum") /num_tgt

    losses = {'loss_ce_dn': loss_ce}

    return losses