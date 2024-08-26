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
from psd2.modeling.reid_heads.box_augmentation import build_box_augmentor
import numpy as np
import cv2
import torch.utils.checkpoint as checkpoint
import copy
import time

@META_ARCH_REGISTRY.register()
class TiRCNN_Baseline(GeneralizedRCNN):
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
        return res

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
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            det_pred_instances, det_losses = self.roi_heads(
                image_list, features, proposals, gt_instances
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
            t0=time.time()
            proposals, _ = self.proposal_generator(image_list, features, None)
            det_pred_instances, _ = self.roi_heads(
                image_list, features, proposals, None
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
        t0=time.time()
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
            t0=time.time()
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
            t0=time.time()
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
class TiRCNN_Baseline_JointDc(TiRCNN_Baseline):
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
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            det_pred_instances, det_losses = self.roi_heads(
                image_list, features, proposals, gt_instances
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
            losses.update(proposal_losses)
            return losses
        else:
            proposals, _ = self.proposal_generator(image_list, features, None)
            det_pred_instances, _ = self.roi_heads(
                image_list, features, proposals, None
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
class TiRCNN_NextBaseline(TiRCNN_Baseline):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        next_res5=nn.Sequential(copy.deepcopy(ret["backbone"].downsample_layers[3]),copy.deepcopy(ret["backbone"].stages[3]),copy.deepcopy(ret["backbone"].norm3))
        bk=ret["backbone"]
        roi_heads=OuterRes5ROIHeads(cfg, bk.output_shape(),next_res5,768)
        ret["roi_heads"]=roi_heads
        return ret

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


from psd2.modeling.backbone.resnet import (
    ResNet,
    BottleneckBlock,
    BasicBlock,
    DeformBottleneckBlock,
)
import torch.nn.init as init
import torch.nn.functional as tF
import copy


@META_ARCH_REGISTRY.register()
class TiRCNN_C4Side(TiRCNN_Baseline):
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
        for p in self.proposal_generator.parameters():
            p.requires_grad_(False)
        for p in self.roi_heads.parameters():
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
class TiRCNN_C4Side_ParaInf(TiRCNN_C4Side):
    def backbone_para(self,x):
        x=self.backbone.stem(x)
        x=self.backbone.res2(x)
        res_d=self.backbone.res3(x)
        res_r=self.side_res3(x)
        alpha = torch.sigmoid(self.alpha_res3)
        res_r = alpha * res_d + (1 - alpha) * res_r
        res_d=self.backbone.res4(res_d)
        res_r=self.side_res4(res_r)
        alpha = torch.sigmoid(self.alpha_res4)
        res_r = alpha * res_d + (1 - alpha) * res_r
        return res_d,res_r
    def get_reid_backbone_features(self, det_backbone_features, image_list):
        return det_backbone_features
        
    def forward(self, input_list):
            assert not self.training
            assert not self.train_task == "det"
            if "query" in input_list[0]:
                return self.inf_query(input_list)
            images, gt_instances = self.preprocess_input(input_list)
            t0=time.time()
            bk_d,bk_r = self.backbone_para(images.tensor)
            # print("det backbone time: {}".format(time.time()-t0))
            det_pred = self.forward_det(images, {"res4":bk_d}, gt_instances)
            reid_feats = self.forward_ps(
                det_pred, bk_r, images, gt_instances
            )
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
            t1=time.time()
            print(t1-t0)
            return outputs

from psd2.modeling.backbone import convnext_tiny
@META_ARCH_REGISTRY.register()
class TiRCNN_NextC4Side(TiRCNN_C4Side):
    @configurable
    def __init__(
        self,
        bn_neck, side_init,conv_next,**kwargs,) -> None:
        super(TiRCNN_C4Side,self).__init__(**kwargs)
        self.side_init = side_init
        self.alpha_res3 = nn.Parameter(torch.tensor(0.0))
        self.alpha_res4 = nn.Parameter(torch.tensor(0.0))
        # side reid network
        self.bn_neck = bn_neck
        # next stage 2
        self.side_res3 = nn.Sequential(copy.deepcopy(conv_next.downsample_layers[1]),copy.deepcopy(conv_next.stages[1]))
        
        self.side_res4 = nn.Sequential(copy.deepcopy(conv_next.downsample_layers[2]),copy.deepcopy(conv_next.stages[2]))
        self.side_res5 = nn.Sequential(copy.deepcopy(conv_next.downsample_layers[3]),copy.deepcopy(conv_next.stages[3]),copy.deepcopy(conv_next.norm3))
        del conv_next
        # remove last stride 
        self.side_res5[0][1].stride=(1,1)
        self.side_res5[0][1].padding=(1,1)
        self.side_res5[0][1].dilation=(2,2)
        # freeze det params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.proposal_generator.parameters():
            p.requires_grad_(False)
        for p in self.roi_heads.parameters():
            p.requires_grad_(False)
    def load_state_dict(self, *args, **kws):
        # NOTE side init is not needed
        output = super(TiRCNN_C4Side,self).load_state_dict(*args, **kws)
        return output

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        next_res5=nn.Sequential(copy.deepcopy(ret["backbone"].downsample_layers[3]),copy.deepcopy(ret["backbone"].stages[3]),copy.deepcopy(ret["backbone"].norm3))
        bk=ret["backbone"]
        roi_heads=OuterRes5ROIHeads(cfg, bk.output_shape(),next_res5,768)
        ret["roi_heads"]=roi_heads
        conv_next=convnext_tiny(cfg,None)
        ret["conv_next"]=conv_next
        return ret
    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        reid_res3_feat = (
            checkpoint.checkpoint(self.side_res3, det_backbone_features["stage1_unorm"])
            if self.use_checkpoint
            else self.side_res3(det_backbone_features["stage1_unorm"])
        )
        alpha = torch.sigmoid(self.alpha_res3)
        reid_res3_feat = (
            alpha * det_backbone_features["stage2_unorm"] + (1 - alpha) * reid_res3_feat
        )
        reid_res4_feat = (
            checkpoint.checkpoint(self.side_res4, reid_res3_feat)
            if self.use_checkpoint
            else self.side_res4(reid_res3_feat)
        )
        del reid_res3_feat
        alpha = torch.sigmoid(self.alpha_res4)
        reid_res4_feat = (
            alpha * det_backbone_features["stage3_unorm"] + (1 - alpha) * reid_res4_feat
        )
        return reid_res4_feat
@META_ARCH_REGISTRY.register()
class TiRCNN_NextC4Side_ParaInf(TiRCNN_NextC4Side):
    def backbone_para(self,x):
        x=self.backbone.downsample_layers[0](x)
        x=self.backbone.stages[0](x)
        res_d=self.backbone.downsample_layers[1](x)
        res_d=self.backbone.stages[1](res_d)
        res_r=self.side_res3(x)
        alpha = torch.sigmoid(self.alpha_res3)
        res_r = alpha * res_d + (1 - alpha) * res_r
        res_d=self.backbone.downsample_layers[2](res_d)
        res_d=self.backbone.stages[2](res_d)
        res_r=self.side_res4(res_r)
        alpha = torch.sigmoid(self.alpha_res4)
        res_r = alpha * res_d + (1 - alpha) * res_r
        return res_d,res_r
    def get_reid_backbone_features(self, det_backbone_features, image_list):
        return det_backbone_features
        
    def forward(self, input_list):
            assert not self.training
            assert not self.train_task == "det"
            if "query" in input_list[0]:
                return self.inf_query(input_list)
            images, gt_instances = self.preprocess_input(input_list)
            t0=time.time()
            bk_d,bk_r = self.backbone_para(images.tensor)
            # print("det backbone time: {}".format(time.time()-t0))
            det_pred = self.forward_det(images, {"stage3_unorm":bk_d}, gt_instances)
            reid_feats = self.forward_ps(
                det_pred, bk_r, images, gt_instances
            )
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
            t1=time.time()
            print(t1-t0)
            return outputs

@META_ARCH_REGISTRY.register()
class TiRCNN_C4Side_JointDc(TiRCNN_Baseline_JointDc,TiRCNN_C4Side):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super(TiRCNN_C4Side,self).__init__(**kwargs)

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

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        reid_res3_feat = (
            checkpoint.checkpoint(self.side_res3, det_backbone_features["res2"].detach())
            if self.use_checkpoint
            else self.side_res3(det_backbone_features["res2"].detach())
        )
        alpha = torch.sigmoid(self.alpha_res3)
        reid_res3_feat = (
            alpha * det_backbone_features["res3"].detach() + (1 - alpha) * reid_res3_feat
        )
        reid_res4_feat = (
            checkpoint.checkpoint(self.side_res4, reid_res3_feat)
            if self.use_checkpoint
            else self.side_res4(reid_res3_feat)
        )
        del reid_res3_feat
        alpha = torch.sigmoid(self.alpha_res4)
        reid_res4_feat = (
            alpha * det_backbone_features["res4"].detach() + (1 - alpha) * reid_res4_feat
        )
        return reid_res4_feat
    
    
@META_ARCH_REGISTRY.register()
class TiRCNN_C4SideTrip_JointDc(TiRCNN_C4Side_JointDc):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.triplet_loss = TripletLoss(0.3, "mean")
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
            reid_loss = self.reid_loss(pos_embs, pos_ids, None)
            oim_lookup = self.reid_loss.lb_layer.lookup_table
            lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=pos_ids.dtype, device=pos_ids.device
            )
            feats1 = pos_embs[pos_ids > -1]
            feats2 = torch.cat([pos_embs[pos_ids > -1], oim_lookup], dim=0)
            trip = self.triplet_loss(
                feats1,
                feats2,
                pos_ids[pos_ids > -1],
                torch.cat([pos_ids[pos_ids > -1], lookup_ids], dim=0),
                normalize_feature=True,
            )
            for k, v in trip.items():
                if "loss" in k:
                    reid_loss[k] = v
                else:
                    get_event_storage().put_scalar(k, v)
            return reid_loss
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

@META_ARCH_REGISTRY.register()
class TiRCNN_C45Side(TiRCNN_C4Side):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha_res5 = nn.Parameter(torch.tensor(0.0))

    def get_reid_person_features(
        self, reid_backbone_features, det_backbone_features, person_boxes
    ):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats_r = self.reid_pooler([reid_backbone_features], d2_boxes)
        roi_feats_r = (
            checkpoint.checkpoint(self.side_res5, roi_feats_r)
            if self.use_checkpoint
            else self.side_res5(roi_feats_r)
        )  # n x c x h x w
        roi_feats_d = self.reid_pooler([det_backbone_features["res4"]], d2_boxes)
        roi_feats_d = self.roi_heads.res5(roi_feats_d)
        alpha = torch.sigmoid(self.alpha_res5)
        return alpha * roi_feats_d + (1 - alpha) * roi_feats_r

    def forward_ps(self, det_pred, det_backbone_features, image_list, gts):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        if self.training:
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples(det_pred, gts)

            pos_featmaps = self.get_reid_person_features(
                reid_bk_feats, det_backbone_features, pos_boxes
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
                reid_bk_feats, det_backbone_features, det_pred["pred_boxes"]
            )
            del reid_bk_feats
            p_embs = self.get_reid_embed(p_featmaps)
            del p_featmaps
            if self.cws:
                cat_scores = torch.cat(det_pred["pred_scores"], dim=0)
                p_embs *= cat_scores  # .unsqueeze(1)
            p_embs = torch.split(p_embs, num_boxes_per_image)
            return p_embs

    def inf_query(self, input_list):
        images, gt_instances = self.preprocess_input(qd["query"] for qd in input_list)
        det_bk_features = self.backbone(images.tensor)
        reid_bk_feats = self.get_reid_backbone_features(det_bk_features, images)
        q_boxes = [gti.gt_boxes.tensor for gti in gt_instances]
        q_featmaps = self.get_reid_person_features(
            reid_bk_feats, det_bk_features, q_boxes
        )
        del det_bk_features
        del reid_bk_feats
        q_embs = self.get_reid_embed(q_featmaps)
        del q_featmaps
        for bi, feat in enumerate(q_embs):
            input_list[bi]["query"]["feat"] = feat
        return input_list


@META_ARCH_REGISTRY.register()
class TiRCNN_C4SideJoint(TiRCNN_C4Side, TiRCNN_Baseline):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        TiRCNN_Baseline.__init__(self, **kwargs)

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

    def forward(self, input_list):
        if self.training:
            images, gt_instances = self.preprocess_input(input_list)
            det_bk_features = self.backbone(images.tensor)
            # forward det
            proposals, proposal_losses = self.proposal_generator(
                images, det_bk_features, gt_instances
            )

            # det_pred_instances, det_losses = self.roi_heads.forward_unms(
            #    images, det_bk_features, proposals, gt_instances
            # )
            det_pred_instances, det_losses = self.roi_heads(
                images, det_bk_features, proposals, gt_instances
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
                        (images, gt_instances),
                        det_bk_features[list(det_bk_features.keys())[-1]],
                        outputs,
                    )
            losses = {}
            losses.update(det_losses)
            losses.update(proposal_losses)
            # forward re-id
            det_pred = {
                "pred_boxes": [],
                "pred_scores": [],
                "pred_pids": [],
            }
            """
            for pred_img in det_pred_instances:
                det_pred["pred_boxes"].append(pred_img.pred_boxes.tensor)
                det_pred["pred_scores"].append(pred_img.scores)
            """
            # back to original pid
            for gti in gt_instances:
                cur_pids = gti.gt_classes
                gti.gt_classes[cur_pids == 0] = -1
                gti.gt_classes[cur_pids > 0] -= 2
            reid_bk_feats = self.get_reid_backbone_features(det_bk_features, images)
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples(
                det_pred, gt_instances
            )

            pos_featmaps = self.get_reid_person_features(
                reid_bk_feats, pos_boxes
            )  # nf x c x h x w

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:

                    self.visualize_training_ps(
                        images.tensor,
                        pos_featmaps.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(pos_featmaps)
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            rl = self.reid_loss(pos_embs, pos_ids, None)
            losses.update(rl)
            return losses
        else:
            return super().forward(input_list)

@META_ARCH_REGISTRY.register()
class TiRCNN_C4HeadJoint(TiRCNN_C4SideJoint):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        TiRCNN_Baseline.__init__(self, **kwargs)
        self.side_init = side_init
        # side reid network
        self.bn_neck = bn_neck
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
    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        return det_backbone_features["res4"]
    def load_state_dict(self, *args, **kws):
        output = TiRCNN_Baseline.load_state_dict(self,*args, **kws)
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
            for si in range(1):
                res_name = "res{}".format(si + 5)
                for pn, pa in state_dict.items():
                    if res_name in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        side_res_params[si][pn.split(res_name)[1][1:]] = pa
                    elif "bn_neck" in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        bn_neck_params[pn.split("bn_neck")[1][1:]] = pa
            self.side_res5.load_state_dict(side_res_params[0])
            if len(bn_neck_params) > 0:
                self.bn_neck.load_state_dict(bn_neck_params)
        return output

from psd2.layers.metric_loss import TripletLoss


@META_ARCH_REGISTRY.register()
class TiRCNN_C4SideTrip1(TiRCNN_C4Side):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.triplet_loss = TripletLoss(0.3, "mean")

    def forward_ps(self, det_pred, det_backbone_features, image_list, gts):
        t0=time.time()
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
                    self.visualize_training_ps(
                        image_list.tensor,
                        pos_featmaps.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(
                pos_featmaps
            )  # before-bnneck is inccorect,for the lookup-table are after-bnneck
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            reid_loss = self.reid_loss(pos_embs, pos_ids, None)
            oim_lookup = self.reid_loss.lb_layer.lookup_table
            lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=pos_ids.dtype, device=pos_ids.device
            )
            feats1 = pos_embs[pos_ids > -1]
            feats2 = torch.cat([pos_embs[pos_ids > -1], oim_lookup], dim=0)
            if feats1.shape[0] < 1:
                reid_loss["loss_triplet"] = torch.zeros(1, device=self.device)
            else:
                trip = self.triplet_loss(
                    feats1,
                    feats2,
                    pos_ids[pos_ids > -1],
                    torch.cat([pos_ids[pos_ids > -1], lookup_ids], dim=0),
                    normalize_feature=True,
                )
                for k, v in trip.items():
                    if "loss" in k:
                        reid_loss[k] = v
                    else:
                        get_event_storage().put_scalar(k, v)
            return reid_loss
        else:
            t0=time.time()
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


@META_ARCH_REGISTRY.register()
class TiRCNN_C4SideSepTrip1(TiRCNN_C4SideTrip1):
    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        reid_res_feat = (
            checkpoint.checkpoint(self.side_res3, det_backbone_features["res2"])
            if self.use_checkpoint
            else self.side_res3(det_backbone_features["res2"])
        )
        reid_res_feat = (
            checkpoint.checkpoint(self.side_res4, reid_res_feat)
            if self.use_checkpoint
            else self.side_res4(reid_res_feat)
        )
        return reid_res_feat




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


from typing import Dict, List, Optional, Tuple
from psd2.structures import Boxes, ImageList, Instances, pairwise_iou


class DnRes5Head(nn.Module):
    """modified from Res5ROIHeads"""

    def __init__(
        self,
        pooler: ROIPooler,
        res5: nn.Module,
        box_predictor: nn.Module,
        proposal_matcher,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            res5 (nn.Sequential): a CNN to compute per-region features, to be used by
                ``box_predictor`` and ``mask_head``. Typically this is a "res5"
                block from a ResNet.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(**kwargs)
        self.pooler = pooler
        if isinstance(res5, (list, tuple)):
            res5 = nn.Sequential(*res5)
        self.res5 = res5
        self.box_predictor = box_predictor
        self.proposal_matcher = proposal_matcher

    def _shared_roi_transform(self, features: torch.Tensor, boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(
        self,
        images: ImageList,
        features: torch.Tensor,
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        del images
        # for training only
        with torch.no_grad():
            # label assign
            all_proposals = []
            for proposals_per_image, targets_per_image in zip(proposals, targets):
                has_gt = len(targets_per_image) > 0
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                matched_idxs, matched_labels = self.proposal_matcher(
                    match_quality_matrix
                )
                pos_mask = matched_labels == 1
                proposals_per_image = proposals_per_image[pos_mask]
                proposals_per_image.gt_classes = torch.zeros(
                    pos_mask.sum(), dtype=torch.long, device=features.device
                )
                if has_gt:
                    sampled_targets = matched_idxs[pos_mask]
                    for trg_name, trg_value in targets_per_image.get_fields().items():
                        if trg_name.startswith("gt_") and not proposals_per_image.has(
                            trg_name
                        ):
                            proposals_per_image.set(
                                trg_name, trg_value[sampled_targets]
                            )
                all_proposals.append(proposals_per_image)
        del targets
        proposal_boxes = [x.proposal_boxes for x in all_proposals]
        box_features = self._shared_roi_transform([features], proposal_boxes)
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        losses = self.box_predictor.losses(predictions, all_proposals)
        losses.pop("loss_cls")
        return losses

from psd2.modeling.roi_heads import Res5ROIHeads
from psd2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from psd2.layers import ShapeSpec

class OuterRes5ROIHeads(Res5ROIHeads):
    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        if self.training:
            ret=checkpoint.checkpoint(self.res5,x)
        else:
            ret=self.res5(x)
        return ret
    @classmethod
    def from_config(cls, cfg, input_shape,res5_module,out_channels):
        # fmt: off
        ret = super(Res5ROIHeads,cls).from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        mask_on           = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )


        ret["res5"] = res5_module
        ret["box_predictor"] = FastRCNNOutputLayers(
                cfg, ShapeSpec(channels=out_channels, height=1, width=1)
            )
        return ret