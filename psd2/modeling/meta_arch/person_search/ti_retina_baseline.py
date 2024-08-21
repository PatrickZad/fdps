from psd2.modeling.meta_arch import RetinaNet
from psd2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
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
import copy

@META_ARCH_REGISTRY.register()
class TiRetina_Baseline(RetinaNet):
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
        det_bk_features = self.backbone.bottom_up(images.tensor)
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
            features = [features[f] for f in self.head_in_features]
            predictions = self.head(features)

            # Transpose the Hi*Wi*A dimension to the middle:
            pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
                predictions, [self.num_classes, 4]
            )
            anchors = self.anchor_generator(features)
            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)

            for gtli in gt_labels:
                gtli[gtli > 1] = 0
            det_losses = self.losses(
                anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    det_pred_instances = []
                    with torch.no_grad():
                        for img_idx, image_size in enumerate(image_list.image_sizes):
                            scores_per_image = [
                                x[img_idx].sigmoid_() for x in pred_logits
                            ]
                            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
                            results_per_image = self.inference_single_image(
                                anchors, scores_per_image, deltas_per_image, image_size
                            )
                            det_pred_instances.append(results_per_image)
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
                        features,
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
            if self.anchor_generator.training:
                self.anchor_generator.eval()
            if self.head.training:
                self.head.eval()
            features = [features[f] for f in self.head_in_features]
            predictions = self.head(features)
            if self.train_with_nms:
                if self.pid_asigner is None:
                    raise NotImplementedError
                    det_pred_instances = None
                    for pred in det_pred_instances:
                        pred_pids = pred.pred_classes
                        pred_pids[pred_pids == 1] = -2
                        pred_pids[pred_pids == 0] = -1
                        pred_pids[pred_pids > 1] -= 2
                else:
                    det_pred_instances = self.forward_inference(
                        image_list, features, predictions
                    )
            else:
                raise NotImplementedError
                if self.pid_asigner is None:
                    det_pred_instances = None
                    for pred in det_pred_instances:
                        pred_pids = pred.pred_classes
                        pred_pids[pred_pids == 1] = -2
                        pred_pids[pred_pids == 0] = -1
                        pred_pids[pred_pids > 1] -= 2
                else:
                    det_pred_instances = None

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
            features = [features[f] for f in self.head_in_features]
            predictions = self.head(features)
            det_pred_instances = self.forward_inference(
                image_list, features, predictions
            )

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
                det_fpn_features = self.backbone(images.tensor)
                return self.forward_det(
                    images, det_fpn_features, gt_instances
                )  # losses
            else:
                if self.backbone.training:
                    self.backbone.eval()
                with torch.no_grad():
                    if not self.train_with_det:
                        det_bk_features = self.backbone.bottom_up(images.tensor)
                        det_fpn_features = None
                    else:
                        det_fpn_features, det_bk_features = self.backbone(
                            images.tensor, return_bk=True
                        )
                    det_pred = self.forward_det(images, det_fpn_features, gt_instances)
                # back to original pid
                for gti in gt_instances:
                    cur_pids = gti.gt_classes
                    gti.gt_classes[cur_pids == 0] = -1
                    gti.gt_classes[cur_pids > 0] -= 2
                return self.forward_ps(det_pred, det_bk_features, images, gt_instances)
        else:
            if self.train_task == "det":
                images, gt_instances = self.preprocess_input(input_list)

                det_fpn_features = self.backbone(images.tensor)

                det_pred = self.forward_det(
                    images, det_fpn_features, gt_instances
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
            det_fpn_features, det_bk_features = self.backbone(
                images.tensor, return_bk=True
            )
            det_pred = self.forward_det(images, det_fpn_features, gt_instances)
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

        threds = [0.05]
        storage = get_event_storage()
        if self.pixel_mean.max() > 1:
            trans_t2img_t = (
                lambda t: t.detach().cpu() * self.pixel_std.cpu().view(-1, 1, 1) / 255.0
                + self.pixel_mean.cpu().view(-1, 1, 1) / 255.0
            )
        else:
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
            if self.input_format == "BGR":
                img_rgb = img_rgb[:, :, ::-1]
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
class TiRetina_Baseline_JointDc(TiRetina_Baseline):
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
            features = [features[f] for f in self.head_in_features]
            predictions = self.head(features)

            # Transpose the Hi*Wi*A dimension to the middle:
            pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
                predictions, [self.num_classes, 4]
            )
            anchors = self.anchor_generator(features)
            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)

            for gtli in gt_labels:
                gtli[gtli > 1] = 0
            det_losses = self.losses(
                anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    det_pred_instances = []
                    with torch.no_grad():
                        for img_idx, image_size in enumerate(image_list.image_sizes):
                            scores_per_image = [
                                x[img_idx].sigmoid_() for x in pred_logits
                            ]
                            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
                            results_per_image = self.inference_single_image(
                                anchors, scores_per_image, deltas_per_image, image_size
                            )
                            det_pred_instances.append(results_per_image)
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
                        features,
                        outputs,
                    )
            losses = {}
            losses.update(det_losses)
            return losses
        else:
            features = [features[f] for f in self.head_in_features]
            predictions = self.head(features)
            det_pred_instances = self.forward_inference(
                image_list, features, predictions
            )

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
            det_fpn_features, det_bk_features = self.backbone(
                            images.tensor, return_bk=True
                        )
            losses= self.forward_det(
                    images, det_fpn_features, gt_instances
                )
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
            det_fpn_features, det_bk_features = self.backbone(
                images.tensor, return_bk=True
            )
            # print("det backbone time: {}".format(time.time()-t0))
            det_pred = self.forward_det(images, det_fpn_features, gt_instances)
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
import torch.utils.checkpoint as checkpoint
from .ti_rcnn_baseline import _load_file


@META_ARCH_REGISTRY.register()
class TiRetina_C4Side(TiRetina_Baseline):
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
        for p in self.anchor_generator.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
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
                try:
                    self.bn_neck.load_state_dict(bn_neck_params, strict=False)
                except Exception as e:
                    print(e)
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
        roi_feats = self.side_res5(roi_feats)  # n x c x h x w
        return roi_feats

    def get_reid_embed(self, p_feat_maps):
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        return tF.normalize(pfeat_embs, dim=-1)

    def vis_forward(self, input_list):
        images, gt_instances = self.preprocess_input(input_list)
        images, gt_instances = self.preprocess_input(input_list)
        det_fpn_features, det_bk_features = self.backbone(images.tensor, return_bk=True)
        det_pred = self.forward_det(images, det_fpn_features, gt_instances)
        # forward ps
        reid_bk_feats = self.get_reid_backbone_features(det_bk_features, images)
        num_boxes_per_image = [bi.shape[0] for bi in det_pred["pred_boxes"]]
        p_featmaps = self.get_reid_person_features(
            reid_bk_feats, det_pred["pred_boxes"]
        )

        p_embs = self.get_reid_embed(p_featmaps)
        del p_featmaps
        p_embs = torch.split(p_embs, num_boxes_per_image)
        # res5
        reid_bk_feats = self.side_res5(reid_bk_feats)
        outputs = []
        for pi, pfeats in enumerate(p_embs):
            org_boxes = det_pred["pred_boxes"][pi].cpu()
            org_scores = det_pred["pred_scores"][pi].cpu().view(-1)
            boxes = org_boxes[org_scores >= 0.3]
            i_feat = reid_bk_feats[pi]
            outputs.append({"p_boxes": boxes, "p_embs": pfeats, "i_feat": i_feat})
        return outputs


@META_ARCH_REGISTRY.register()
class TiRetina_C4Side_JointDc(TiRetina_Baseline_JointDc,TiRetina_C4Side):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super(TiRetina_C4Side,self).__init__(**kwargs)

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
class TiRetina_C4SideTrip_JointDc(TiRetina_C4Side_JointDc):
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

from psd2.layers.metric_loss import TripletLoss


@META_ARCH_REGISTRY.register()
class TiRetina_C4SideTrip1(TiRetina_C4Side):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.triplet_loss = TripletLoss(0.3, "mean")

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
            return p_embs


@META_ARCH_REGISTRY.register()
class TiRetina_C4SideTrip1Ms1(TiRetina_C4SideTrip1):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.side_res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [2, 1, 1],
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
        self.lateral3 = Conv2d(
            512,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            norm=None,
        )
        self.lateral4 = Conv2d(
            1024,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            norm=None,
        )
        self.lateral5 = Conv2d(
            2048,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            norm=None,
        )
        self.output3 = Conv2d(
            256,
            256,
            kernel_size=1,
            bias=True,
            norm=None,
        )
        self.output4 = Conv2d(
            256,
            256,
            kernel_size=1,
            bias=True,
            norm=None,
        )
        self.output5 = Conv2d(
            256,
            256,
            kernel_size=1,
            bias=True,
            norm=None,
        )
        self.reid_pooler3 = ROIPooler(
            output_size=self.reid_pooler.output_size,
            scales=[1 / 8],
            sampling_ratio=2,
            pooler_type="ROIAlignV2",
        )
        self.reid_pooler5 = ROIPooler(
            output_size=self.reid_pooler.output_size,
            scales=[1 / 32],
            sampling_ratio=2,
            pooler_type="ROIAlignV2",
        )
        self.pfeat_pooling3 = copy.deepcopy(self.pfeat_pooling)
        self.pfeat_pooling5 = copy.deepcopy(self.pfeat_pooling)
        self.bn_neck3 = copy.deepcopy(self.bn_neck)
        self.bn_neck5 = copy.deepcopy(self.bn_neck)
        self.reid_loss3 = copy.deepcopy(self.reid_loss)
        self.reid_loss5 = copy.deepcopy(self.reid_loss)

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
        alpha = torch.sigmoid(self.alpha_res4)
        reid_res4_feat = (
            alpha * det_backbone_features["res4"] + (1 - alpha) * reid_res4_feat
        )
        reid_res5_feat = self.side_res5(reid_res4_feat)

        reid_res3_feat = self.lateral3(reid_res3_feat)
        reid_res4_feat = self.lateral4(reid_res4_feat)
        reid_res5_feat = self.lateral5(reid_res5_feat)
        return [reid_res3_feat, reid_res4_feat, reid_res5_feat]

    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats3 = self.reid_pooler3([reid_backbone_features[0]], d2_boxes)
        roi_feats3 = self.output3(roi_feats3)
        roi_feats4 = self.reid_pooler([reid_backbone_features[1]], d2_boxes)
        roi_feats4 = self.output4(roi_feats4)
        roi_feats5 = self.reid_pooler5([reid_backbone_features[2]], d2_boxes)
        roi_feats5 = self.output5(roi_feats5)
        return [roi_feats3, roi_feats4, roi_feats5]

    def get_reid_embed(self, p_feat_maps):
        pfeat_embs3 = self.pfeat_pooling3(p_feat_maps[0])
        pfeat_embs3 = pfeat_embs3.view(pfeat_embs3.shape[0], pfeat_embs3.shape[1])
        pfeat_embs3 = self.bn_neck3(pfeat_embs3)
        pfeat_embs4 = self.pfeat_pooling(p_feat_maps[1])
        pfeat_embs4 = pfeat_embs4.view(pfeat_embs4.shape[0], pfeat_embs4.shape[1])
        pfeat_embs4 = self.bn_neck(pfeat_embs4)
        pfeat_embs5 = self.pfeat_pooling5(p_feat_maps[2])
        pfeat_embs5 = pfeat_embs5.view(pfeat_embs5.shape[0], pfeat_embs5.shape[1])
        pfeat_embs5 = self.bn_neck5(pfeat_embs5)
        if self.training:
            return pfeat_embs3, pfeat_embs4, pfeat_embs5
        else:
            return tF.normalize(
                torch.cat([pfeat_embs3, pfeat_embs4, pfeat_embs5], dim=-1), dim=-1
            )

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
                        pos_featmaps[-1].split(
                            [bxs.shape[0] for bxs in pos_boxes], dim=0
                        ),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(pos_featmaps)
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            losses = {}
            losses["loss_oim_3"] = self.reid_loss3(pos_embs[0], pos_ids, None)[
                "loss_oim"
            ]
            oim_lookup = self.reid_loss3.lb_layer.lookup_table
            lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=pos_ids.dtype, device=pos_ids.device
            )
            feats1 = pos_embs[0][pos_ids > -1]
            feats2 = torch.cat([pos_embs[0][pos_ids > -1], oim_lookup], dim=0)
            trip = self.triplet_loss(
                feats1,
                feats2,
                pos_ids[pos_ids > -1],
                torch.cat([pos_ids[pos_ids > -1], lookup_ids], dim=0),
                normalize_feature=True,
            )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k + "_3"] = v
            losses["loss_oim_4"] = self.reid_loss(pos_embs[1], pos_ids, None)[
                "loss_oim"
            ]
            oim_lookup = self.reid_loss.lb_layer.lookup_table
            lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=pos_ids.dtype, device=pos_ids.device
            )
            feats1 = pos_embs[1][pos_ids > -1]
            feats2 = torch.cat([pos_embs[1][pos_ids > -1], oim_lookup], dim=0)
            trip = self.triplet_loss(
                feats1,
                feats2,
                pos_ids[pos_ids > -1],
                torch.cat([pos_ids[pos_ids > -1], lookup_ids], dim=0),
                normalize_feature=True,
            )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k + "_4"] = v
            losses["loss_oim_5"] = self.reid_loss5(pos_embs[2], pos_ids, None)[
                "loss_oim"
            ]
            oim_lookup = self.reid_loss5.lb_layer.lookup_table
            lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=pos_ids.dtype, device=pos_ids.device
            )
            feats1 = pos_embs[2][pos_ids > -1]
            feats2 = torch.cat([pos_embs[2][pos_ids > -1], oim_lookup], dim=0)
            trip = self.triplet_loss(
                feats1,
                feats2,
                pos_ids[pos_ids > -1],
                torch.cat([pos_ids[pos_ids > -1], lookup_ids], dim=0),
                normalize_feature=True,
            )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k + "_5"] = v
            return losses
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


@META_ARCH_REGISTRY.register()
class TiRetina_C4SideTrip1Ms2(TiRetina_C4SideTrip1Ms1):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output3 = Conv2d(
            256,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            norm=None,
        )
        self.output4 = Conv2d(
            256,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            norm=None,
        )
        self.output5 = Conv2d(
            256,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            norm=None,
        )


@META_ARCH_REGISTRY.register()
class TiRetina_C4SideD3(TiRetina_C4Side):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super(TiRetina_C4Side, self).__init__(**kwargs)

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
                    "block_class": DeformBottleneckBlock,
                    "deform_modulated": True,
                    "deform_num_groups": 1,
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
                    "block_class": DeformBottleneckBlock,
                    "deform_modulated": True,
                    "deform_num_groups": 1,
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
                    "block_class": DeformBottleneckBlock,
                    "deform_modulated": True,
                    "deform_num_groups": 1,
                }
            )
        )
        # freeze det params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.anchor_generator.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
            p.requires_grad_(False)

    def load_state_dict(self, *args, **kws):
        output = super(TiRetina_C4Side, self).load_state_dict(*args, **kws)
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
            self.side_res3.load_state_dict(side_res_params[0], strict=False)
            self.side_res4.load_state_dict(side_res_params[1], strict=False)
            self.side_res5.load_state_dict(side_res_params[2], strict=False)
            if len(bn_neck_params) > 0:
                try:
                    self.bn_neck.load_state_dict(bn_neck_params, strict=False)
                except Exception as e:
                    print(e)
        return output

    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats = self.reid_pooler([reid_backbone_features], d2_boxes)
        roi_feats =checkpoint.checkpoint(self.side_res5,roi_feats)  # n x c x h x w
        return roi_feats

@META_ARCH_REGISTRY.register()
class TiRetina_R34C4Side(TiRetina_C4Side):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super(TiRetina_C4Side, self).__init__(**kwargs)

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
                    "in_channels": 64,
                    "out_channels": 128,
                    "norm": "BN",
                    "block_class": BasicBlock,
                }
            )
        )
        self.side_res4 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 6,
                    "stride_per_block": [2, 1, 1, 1, 1, 1],
                    "in_channels": 128,
                    "out_channels": 256,
                    "norm": "BN",
                    "block_class": BasicBlock,
                }
            )
        )
        self.side_res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 256,
                    "out_channels": 512,
                    "norm": "BN",
                    "block_class": BasicBlock,
                }
            )
        )
        # freeze det params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.anchor_generator.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
            p.requires_grad_(False)


@META_ARCH_REGISTRY.register()
class TiRetina_R34C4SideTrip1(TiRetina_R34C4Side):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.triplet_loss = TripletLoss(0.3, "mean")

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
            if feats1.shape[0] > 0:
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
            else:
                reid_loss["loss_triplet"] = torch.zeros(1, device=self.device)
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
            return p_embs


@META_ARCH_REGISTRY.register()
class TiRetina_R34C4Ladder3(TiRetina_Baseline):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super().__init__(**kwargs)

        self.side_init = side_init
        bk_out_shape = self.backbone.bottom_up.output_shape()
        self.lateral_conv2 = Conv2d(
            bk_out_shape["res2"].channels,
            64,
            kernel_size=1,
            bias=False,
        )  # same with fpn
        self.lateral_conv3 = Conv2d(
            bk_out_shape["res3"].channels,
            128,
            kernel_size=1,
            bias=False,
        )  # same with fpn
        self.lateral_conv4 = Conv2d(
            bk_out_shape["res4"].channels,
            256,
            kernel_size=1,
            bias=False,
        )  # same with fpn
        # side reid network
        self.bn_neck = bn_neck
        self.side_res3 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 4,
                    "stride_per_block": [2, 1, 1, 1],
                    "in_channels": 64,
                    "out_channels": 128,
                    "norm": "BN",
                    "block_class": BasicBlock,
                }
            )
        )
        self.side_res4 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 6,
                    "stride_per_block": [2, 1, 1, 1, 1, 1],
                    "in_channels": 128,
                    "out_channels": 256,
                    "norm": "BN",
                    "block_class": BasicBlock,
                }
            )
        )
        self.side_res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 256,
                    "out_channels": 512,
                    "norm": "BN",
                    "block_class": BasicBlock,
                }
            )
        )
        # freeze det params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.anchor_generator.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
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
            lateral_2_params = {}
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
                    elif "down_channel" in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        lateral_2_params[pn.split("down_channel")[1][1:]] = pa
            self.side_res3.load_state_dict(side_res_params[0])
            self.side_res4.load_state_dict(side_res_params[1])
            self.side_res5.load_state_dict(side_res_params[2])
            if len(bn_neck_params) > 0:
                self.bn_neck.load_state_dict(bn_neck_params)
            if len(lateral_2_params):
                self.lateral_conv2.load_state_dict(lateral_2_params)
        return output

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        reid_res3_feat = self.side_res3(
            self.lateral_conv2(det_backbone_features["res2"])
        )
        reid_res3_feat = (
            self.lateral_conv3(det_backbone_features["res3"]) + reid_res3_feat
        )
        reid_res4_feat = self.side_res4(reid_res3_feat)
        del reid_res3_feat
        reid_res4_feat = (
            self.lateral_conv4(det_backbone_features["res4"]) + reid_res4_feat
        )
        return reid_res4_feat

    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats = self.reid_pooler([reid_backbone_features], d2_boxes)
        return roi_feats

    def get_reid_embed(self, p_feat_maps):
        p_feat_maps = self.side_res5(p_feat_maps)  # n x c x h x w
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        return tF.normalize(pfeat_embs, dim=-1)


from psd2.modeling.backbone.osnet import OSBlock, OSNet, Conv1x1


@META_ARCH_REGISTRY.register()
class TiRetina_Os4Ladder3(TiRetina_Baseline):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super().__init__(**kwargs)

        self.side_init = side_init
        bk_out_shape = self.backbone.bottom_up.output_shape()
        self.lateral_conv2 = Conv2d(
            bk_out_shape["res2"].channels,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )  # downsample
        self.lateral_conv3 = Conv2d(
            bk_out_shape["res3"].channels,
            384,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )  # downsample
        self.lateral_conv4 = Conv2d(
            bk_out_shape["res4"].channels,
            512,
            kernel_size=1,
            bias=False,
        )  # same with fpn
        # side reid network
        self.bn_neck = bn_neck
        self.side_conv3 = OSNet._make_layer(
            block=OSBlock,
            layer=2,
            in_channels=256,
            out_channels=384,
            reduce_spatial_size=True,
            IN=False,
        )

        self.side_conv4 = OSNet._make_layer(
            block=OSBlock,
            layer=2,
            in_channels=384,
            out_channels=512,
            reduce_spatial_size=False,
            IN=False,
        )  # different from original osnet on the downsampling option
        self.side_conv5 = Conv1x1(512, 512)

        # init side network
        for osm in [self.side_conv3, self.side_conv4, self.side_conv5]:
            for m in osm.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # freeze det params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.anchor_generator.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
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
            if "side_conv" in param_k:
                model_resume = False
                break
        if not model_resume and self.side_init != "":
            # model is not resuming from saved checkpoint
            # init side networks
            state_dict = _load_file(self.side_init)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            side_conv_params = [{}, {}, {}]
            bn_neck_params = {}
            lateral2_params = {}
            for si in range(3):
                conv_name = "conv{}.".format(si + 3)
                for pn, pa in state_dict.items():
                    if pn.startswith(conv_name):
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        npn = pn[6:]
                        side_conv_params[si][npn] = pa
                    elif "bn_neck" in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        bn_neck_params[pn.split("bn_neck")[1][1:]] = pa
                    elif pn.startswith("down_channel"):
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        lateral2_params[pn.split("down_channel")[1][1:]] = pa
            self.side_conv3.load_state_dict(side_conv_params[0])
            self.side_conv4.load_state_dict(side_conv_params[1], strict=False)
            self.side_conv5.load_state_dict(side_conv_params[2])
            if len(bn_neck_params) > 0:
                self.bn_neck.load_state_dict(bn_neck_params)
            if len(lateral2_params) > 0:
                self.lateral_conv2.load_state_dict(lateral2_params)

        return output

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list

        reid_conv3_feat = self.side_conv3(
            self.lateral_conv2(det_backbone_features["res2"])
        )

        reid_conv4_feat = self.side_conv4(
            self.lateral_conv3(det_backbone_features["res3"]) + reid_conv3_feat
        )
        reid_conv4_feat = (
            self.lateral_conv4(det_backbone_features["res4"]) + reid_conv4_feat
        )

        return reid_conv4_feat

    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats = self.reid_pooler([reid_backbone_features], d2_boxes)
        return roi_feats

    def get_reid_embed(self, p_feat_maps):
        p_feat_maps = self.side_conv5(p_feat_maps)  # n x c x h x w
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        return tF.normalize(pfeat_embs, dim=-1)


from psd2.layers import batched_nms


@META_ARCH_REGISTRY.register()
class TiRetina_Coupled_bk(TiRetina_Baseline):
    @configurable
    def __init__(self, bn_neck, **kwargs) -> None:
        super().__init__(**kwargs)

        # side reid network
        self.bn_neck = bn_neck

    @classmethod
    def from_config(cls, cfg):
        res = super().from_config(cfg)
        bn_neck = nn.BatchNorm1d(cfg.REID_HEAD.PERSON_FEATURE.DIM)
        init.constant_(bn_neck.bias, 0.0)
        init.normal_(bn_neck.weight, std=0.01)
        bn_neck.bias.requires_grad_(False)
        res["bn_neck"] = bn_neck
        return res

    def forward(self, input_list):
        if self.training:
            images, gt_instances = self.preprocess_input(input_list)
            bk_features = self.backbone(images.tensor)
            features = [bk_features[f] for f in self.head_in_features]
            det_pred = self.head(features)
            pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
                det_pred, [self.num_classes, 4]
            )
            anchors = self.anchor_generator(features)
            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            det_gt_labels = []
            for gtl in gt_labels:
                det_gtl = gtl.clone()
                det_gtl[det_gtl > 1] = 0
                det_gt_labels.append(det_gtl)
            losses = self.losses(
                anchors, pred_logits, det_gtl, pred_anchor_deltas, gt_boxes
            )
            feat_embed = [f.flatten(2) for f in features]
            feat_embed = (
                torch.cat(feat_embed, dim=2).transpose(1, 2).unsqueeze(2)
            )  # n x hw x 1 x c
            feat_embed = feat_embed.expand(-1, -1, self.head.num_anchors, -1)
            feat_embed = feat_embed.view(
                feat_embed.shape[0], -1, feat_embed.shape[-1]
            )  # n x hwa x c
            pid_labels = torch.stack(gt_labels)  # n x r
            pid_labels[pid_labels == 0] = -1
            pid_labels[pid_labels == 1] = -2
            pid_labels[pid_labels > 1] -= 2
            pos_mask = pid_labels > -2
            pos_embeds = feat_embed[pos_mask]
            pos_embeds = self.bn_neck(pos_embeds)
            rl = self.reid_loss(pos_embeds, pid_labels[pos_mask], None)
            losses.update(rl)
            return losses
        else:
            if "query" in input_list[0]:
                return self.inf_query(input_list)
            images, gt_instances = self.preprocess_input(input_list)
            bk_features = self.backbone(images.tensor)
            features = [bk_features[f] for f in self.head_in_features]
            det_pred = self.head(features)
            pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
                det_pred, [self.num_classes, 4]
            )
            anchors = self.anchor_generator(features)
            feat_embed = [f.flatten(2) for f in features]
            feat_embed = (
                torch.cat(feat_embed, dim=2).transpose(1, 2).unsqueeze(2)
            )  # n x hw x 1 x c
            feat_embed = feat_embed.expand(
                -1, -1, self.head.num_anchors, -1
            )  # n x hw x a x c
            feat_embed = feat_embed.reshape(
                (feat_embed.shape[0], -1, feat_embed.shape[-1])
            )  # n x hwa x c
            results = []
            reid_feats = []
            for img_idx, image_size in enumerate(images.image_sizes):
                scores_per_image = [x[img_idx].sigmoid_() for x in pred_logits]
                deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
                results_per_image, reid_feats_i = self.inference_single_image(
                    anchors,
                    scores_per_image,
                    deltas_per_image,
                    image_size,
                    feat_embed[img_idx],
                )
                results.append(results_per_image)
                reid_feats.append(reid_feats_i)
            det_pred = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pi, pred_img in enumerate(results):
                boxes = pred_img.pred_boxes.tensor
                scores = pred_img.scores
                det_pred["pred_boxes"].append(boxes)
                det_pred["pred_scores"].append(scores.unsqueeze(1))
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
        return input_list

    def inference_single_image(
        self,
        anchors,
        box_cls,
        box_delta,
        image_size,
        feat_embs,
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        pred = self._decode_multi_level_predictions(
            anchors,
            box_cls,
            box_delta,
            self.test_score_thresh,
            self.test_topk_candidates,
            image_size,
        )
        keep = batched_nms(  # per-class NMS
            pred.pred_boxes.tensor, pred.scores, pred.pred_classes, self.test_nms_thresh
        )
        return (
            pred[keep[: self.max_detections_per_image]],
            feat_embs[keep[: self.max_detections_per_image]],
        )

    def _decode_per_level_predictions(
        self,
        anchors,
        pred_scores,
        pred_deltas,
        score_thresh,
        topk_candidates,
        image_size,
    ) -> Instances:
        """
        Decode boxes and classification predictions of one featuer level, by
        the following steps:
        1. filter the predictions based on score threshold and top K scores.
        2. transform the box regression outputs
        3. return the predicted scores, classes and boxes

        Args:
            anchors: Boxes, anchor for this feature level
            pred_scores: HxWxA,K
            pred_deltas: HxWxA,4

        Returns:
            Instances: with field "scores", "pred_boxes", "pred_classes".
        """
        # Apply two filtering to make NMS faster.
        # 1. Keep boxes with confidence score higher than threshold
        keep_idxs = pred_scores > score_thresh
        pred_scores = pred_scores[keep_idxs]
        topk_idxs = torch.nonzero(keep_idxs)  # Kx2

        # 2. Keep top k top scoring boxes only
        num_topk = min(topk_candidates, topk_idxs.size(0))
        pred_scores, idxs = pred_scores.topk(num_topk)
        topk_idxs = topk_idxs[idxs]

        anchor_idxs, classes_idxs = topk_idxs.unbind(dim=1)

        pred_boxes = self.box2box_transform.apply_deltas(
            pred_deltas[anchor_idxs], anchors.tensor[anchor_idxs]
        )
        return Instances(
            image_size,
            pred_boxes=Boxes(pred_boxes),
            scores=pred_scores,
            pred_classes=classes_idxs,
        )

    def _decode_multi_level_predictions(
        self,
        anchors,
        pred_scores,
        pred_deltas,
        score_thresh,
        topk_candidates,
        image_size,
    ) -> Instances:
        """
        Run `_decode_per_level_predictions` for all feature levels and concat the results.
        """
        predictions = [
            self._decode_per_level_predictions(
                anchors_i,
                box_cls_i,
                box_reg_i,
                self.test_score_thresh,
                self.test_topk_candidates,
                image_size,
            )
            # Iterate over every feature level
            for box_cls_i, box_reg_i, anchors_i in zip(
                pred_scores, pred_deltas, anchors
            )
        ]
        return predictions[0].cat(
            predictions
        )  # 'Instances.cat' is not scriptale but this is


import torch.utils.checkpoint as checkpoint


@META_ARCH_REGISTRY.register()
class TiRetina_C4Res5Joint(TiRetina_Baseline):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super().__init__(**kwargs)

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

            side_res_params = {}
            bn_neck_params = {}
            res_name = "res5"
            for pn, pa in state_dict.items():
                if res_name in pn:
                    if not isinstance(pa, torch.Tensor):
                        pa = torch.from_numpy(pa).to(self.device)
                    side_res_params[pn.split(res_name)[1][1:]] = pa
                elif "bn_neck" in pn:
                    if not isinstance(pa, torch.Tensor):
                        pa = torch.from_numpy(pa).to(self.device)
                    bn_neck_params[pn.split("bn_neck")[1][1:]] = pa
            self.side_res5.load_state_dict(side_res_params)
            if len(bn_neck_params) > 0:
                try:
                    self.bn_neck.load_state_dict(bn_neck_params, strict=False)
                except Exception as e:
                    print(e)
        return output

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        return det_backbone_features["res4"]

    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats = self.reid_pooler([reid_backbone_features], d2_boxes)
        return roi_feats

    def get_reid_embed(self, p_feat_maps):
        roi_feats = checkpoint.checkpoint(self.side_res5, p_feat_maps)  # n x c x h x w
        pfeat_embs = self.pfeat_pooling(roi_feats)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        if self.training:
            return pfeat_embs
        else:
            return tF.normalize(pfeat_embs, dim=-1)

    def forward(self, input_list):
        if self.training:
            images, gt_instances = self.preprocess_input(input_list)
            det_fpn_features, det_bk_features = self.backbone(
                images.tensor, return_bk=True
            )
            features = [det_fpn_features[f] for f in self.head_in_features]
            predictions = self.head(features)

            # Transpose the Hi*Wi*A dimension to the middle:
            pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
                predictions, [self.num_classes, 4]
            )
            anchors = self.anchor_generator(features)
            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            for gtli in gt_labels:
                gtli[gtli > 1] = 0
            losses = self.losses(
                anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
            # back to original pid
            for gti in gt_instances:
                cur_pids = gti.gt_classes
                gti.gt_classes[cur_pids == 0] = -1
                gti.gt_classes[cur_pids > 0] -= 2
            det_pred = {
                "pred_boxes": [],
                "pred_scores": [],
                "pred_pids": [],
            }
            rl = self.forward_ps(det_pred, det_bk_features, images, gt_instances)
            losses.update(rl)
            return losses
        else:
            return super().forward(input_list)


@META_ARCH_REGISTRY.register()
class TiRetina_Coupled(TiRetina_Baseline):
    @configurable
    def __init__(self, bn_neck, *args, **kws) -> None:
        super().__init__(*args, **kws)
        self.num_anchor = self.anchor_generator.num_cell_anchors[0]
        self.bn_neck = bn_neck

    @classmethod
    def from_config(cls, cfg):
        res = super().from_config(cfg)
        bn_neck = nn.BatchNorm1d(cfg.REID_HEAD.PERSON_FEATURE.DIM)
        init.constant_(bn_neck.bias, 0.0)
        init.normal_(bn_neck.weight, std=0.01)
        bn_neck.bias.requires_grad_(False)
        res["bn_neck"] = bn_neck
        return res

    def inf_query(self, input_list):
        return input_list

    def forward(self, input_list):
        if "query" in input_list[0]:
            return self.inf_query(input_list)
        images, gt_instances = self.preprocess_input(input_list)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        predictions = self.head(features)
        if self.training:
            # Transpose the Hi*Wi*A dimension to the middle:
            pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
                predictions, [self.num_classes, 4]
            )
            anchors = self.anchor_generator(features)
            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)

            features_anchors = [
                feat.unsqueeze(1)
                .expand(-1, self.num_anchor, -1, -1, -1)
                .permute(0, 3, 4, 1, 2)
                .reshape(feat.shape[0], -1, feat.shape[1])
                for feat in features
            ]
            features_anchors = torch.cat(features_anchors, dim=1)
            gt_pids = torch.stack(gt_labels).clone()
            gt_pids[gt_pids == 1] = -2
            gt_pids[gt_pids == 0] = -1
            gt_pids[gt_pids > 1] -= 2
            reid_loss = self.reid_loss(
                self.bn_neck(features_anchors[gt_pids > -2]),
                gt_pids[gt_pids > -2],
                None,
            )

            for gtli in gt_labels:
                gtli[gtli > 1] = 0
            det_losses = self.losses(
                anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    det_pred_instances = []
                    with torch.no_grad():
                        for img_idx, image_size in enumerate(images.image_sizes):
                            features_per_image = [
                                x[img_idx]
                                .unsqueeze(0)
                                .repeat(self.num_anchor, 1, 1, 1)
                                .permute(2, 3, 0, 1)
                                .flatten(0, 2)
                                for x in features
                            ]  # list of ( H x W x A, C)
                            scores_per_image = [
                                x[img_idx].sigmoid_().detach() for x in pred_logits
                            ]
                            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
                            results_per_image = self.inference_single_image(
                                anchors,
                                scores_per_image,
                                deltas_per_image,
                                features_per_image,
                                image_size,
                            )
                            det_pred_instances.append(results_per_image)
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
                        features,
                        outputs,
                    )
            losses = {}
            losses.update(det_losses)
            losses.update(reid_loss)
            return losses
        else:
            pred_instances = self.forward_inference(images, features, predictions)
            outputs = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pi, inst in enumerate(pred_instances):
                boxes = inst.pred_boxes.tensor
                org_boxes = _resize_boxes(
                    boxes, images.image_sizes[pi], gt_instances[pi]._org_hw
                )
                scores = inst.scores.unsqueeze(1)
                outputs["pred_boxes"].append(org_boxes)
                outputs["pred_scores"].append(scores)
                pred_embs = self.bn_neck(inst.pred_embs)
                pred_embs = tF.normalize(pred_embs, dim=-1)
                if self.cws:
                    outputs["reid_feats"].append(pred_embs * scores)
                else:
                    outputs["reid_feats"].append(pred_embs)
            return outputs

    def forward_inference(self, images, features, predictions):
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4]
        )
        anchors = self.anchor_generator(features)
        results = []
        for img_idx, image_size in enumerate(images.image_sizes):
            features_per_image = [
                x[img_idx]
                .unsqueeze(0)
                .repeat(self.num_anchor, 1, 1, 1)
                .permute(2, 3, 0, 1)
                .flatten(0, 2)
                for x in features
            ]  # list of ( H x W x A, C)
            scores_per_image = [
                x[img_idx].sigmoid_() for x in pred_logits
            ]  # list of (H x W x A, 1)
            deltas_per_image = [
                x[img_idx] for x in pred_anchor_deltas
            ]  # list of (H x W x A, 4)
            results_per_image = self.inference_single_image(
                anchors,
                scores_per_image,
                deltas_per_image,
                features_per_image,
                image_size,
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
        self,
        anchors,
        box_cls,
        box_delta,
        box_emb,
        image_size,
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        pred = self._decode_multi_level_predictions(
            anchors,
            box_cls,
            box_delta,
            box_emb,
            self.test_score_thresh,
            self.test_topk_candidates,
            image_size,
        )
        keep = batched_nms(  # per-class NMS
            pred.pred_boxes.tensor, pred.scores, pred.pred_classes, self.test_nms_thresh
        )
        return pred[keep[: self.max_detections_per_image]]

    def _decode_per_level_predictions(
        self,
        anchors,
        pred_scores,
        pred_deltas,
        pred_embs,
        score_thresh,
        topk_candidates,
        image_size,
    ) -> Instances:
        """
        Decode boxes and classification predictions of one featuer level, by
        the following steps:
        1. filter the predictions based on score threshold and top K scores.
        2. transform the box regression outputs
        3. return the predicted scores, classes and boxes

        Args:
            anchors: Boxes, anchor for this feature level
            pred_scores: HxWxA,K
            pred_deltas: HxWxA,4

        Returns:
            Instances: with field "scores", "pred_boxes", "pred_classes".
        """
        # Apply two filtering to make NMS faster.
        # 1. Keep boxes with confidence score higher than threshold
        keep_idxs = pred_scores > score_thresh
        pred_scores = pred_scores[keep_idxs]
        topk_idxs = torch.nonzero(keep_idxs)  # Kx2

        # 2. Keep top k top scoring boxes only
        num_topk = min(topk_candidates, topk_idxs.size(0))
        pred_scores, idxs = pred_scores.topk(num_topk)
        topk_idxs = topk_idxs[idxs]

        anchor_idxs, classes_idxs = topk_idxs.unbind(dim=1)

        pred_boxes = self.box2box_transform.apply_deltas(
            pred_deltas[anchor_idxs], anchors.tensor[anchor_idxs]
        )
        return Instances(
            image_size,
            pred_boxes=Boxes(pred_boxes),
            scores=pred_scores,
            pred_classes=classes_idxs,
            pred_embs=pred_embs[anchor_idxs],
        )

    def _decode_multi_level_predictions(
        self,
        anchors,
        pred_scores,
        pred_deltas,
        pred_embs,
        score_thresh,
        topk_candidates,
        image_size,
    ) -> Instances:
        """
        Run `_decode_per_level_predictions` for all feature levels and concat the results.
        """
        predictions = [
            self._decode_per_level_predictions(
                anchors_i,
                box_cls_i,
                box_reg_i,
                box_emb_i,
                self.test_score_thresh,
                self.test_topk_candidates,
                image_size,
            )
            # Iterate over every feature level
            for box_cls_i, box_reg_i, box_emb_i, anchors_i in zip(
                pred_scores, pred_deltas, pred_embs, anchors
            )
        ]
        return predictions[0].cat(
            predictions
        )  # 'Instances.cat' is not scriptale but this is
