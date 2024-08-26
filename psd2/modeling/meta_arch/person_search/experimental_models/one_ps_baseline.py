import torch
from ..build import META_ARCH_REGISTRY
from psd2.config import configurable
from .base import SearchBase
from psd2.layers.extra_det_head import CenterNetHead, FCOSHead, RetinaHead
from psd2.modeling.matcher import MinCostMatcher as Matcher
from psd2.layers.set_criterion import SrcnnSetCriterion as SetCriterion
from psd2.modeling.reid_heads import build_reid_head
from psd2.structures.nested_tensor import nested_collate_fn_idvi as nested_collate_fn
from psd2.structures.nested_tensor import NestedTensor
from psd2.utils.events import get_event_storage
from psd2.utils.visualizer import Visualizer, mlvl_pca_feat


@META_ARCH_REGISTRY.register()
class OnePS_Baseline(SearchBase):
    @configurable
    def __init__(self, cfg, pre_define, in_features, criterion, reid_head):
        super().__init__(cfg)

        self.num_boxes = cfg.TEST.DETECTIONS_PER_IMAGE
        self.det_head_type = cfg.DETECTOR.MODEL.HEAD.NAME
        self.in_features = in_features
        self.reid_head = reid_head
        self.criterion = criterion
        self.pre_define = pre_define
        self.cws = cfg.CWS
        self.inf_topk = 100

        # Build Head.
        det_head_type = cfg.DETECTOR.MODEL.HEAD.NAME
        if det_head_type == "center":
            self.det_head = CenterNetHead(
                cfg=cfg, backbone_shape=self.backbone.output_shape()
            )
        elif det_head_type == "retina":
            backbone_shape = self.backbone.output_shape()
            feature_shapes = [backbone_shape[f] for f in in_features]
            self.det_head = RetinaHead(cfg=cfg, feature_shapes=feature_shapes)
        elif det_head_type == "fcos":
            self.det_head = FCOSHead(cfg=cfg)
        else:
            raise NotImplementedError

    @classmethod
    def from_config(cls, cfg):
        dt_cfg = cfg.DETECTOR
        model_cfg = dt_cfg.MODEL
        in_features = model_cfg.IN_FEATURES
        det_loss_cfg = dt_cfg.LOSS
        head_cfg = model_cfg.HEAD

        # Loss parameters:
        loss_wgts = det_loss_cfg.LOSS_WEIGHTS
        class_weight = loss_wgts.CLS
        giou_weight = loss_wgts.BOX_GIOU
        l1_weight = loss_wgts.BOX_L1
        weight_dict = {
            "loss_ce": class_weight,
            "loss_bbox": l1_weight,
            "loss_giou": giou_weight,
        }

        # Build Criterion.
        matcher = Matcher(
            model_cfg.HEAD.NAME,
            det_loss_cfg.FOCAL.ALPHA,
            det_loss_cfg.FOCAL.GAMMA,
            model_cfg.PRE_DEFINE,
            head_cfg.OBJECT_SIZES_OF_INTEREST,
            cost_class=class_weight,
            cost_bbox=l1_weight,
            cost_giou=giou_weight,
        )

        losses = ["labels", "boxes"]

        criterion = SetCriterion(
            num_classes=1,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=loss_wgts.NO_OBJECT,
            losses=losses,
            use_focal=True,
            focal_alpha=det_loss_cfg.FOCAL.ALPHA,
            focal_gamma=det_loss_cfg.FOCAL.GAMMA,
        )
        reid_head = build_reid_head(cfg.REID_HEAD)
        return {
            "cfg": cfg,
            "pre_define": model_cfg.PRE_DEFINE,
            "in_features": in_features,
            "criterion": criterion,
            "reid_head": reid_head,
        }

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
            img_pids.append(torch.tensor(input_dict["ids"], device=self.device))
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

    def forward(self, input_list):
        if "query" in input_list[0]:
            return self.inf_query(input_list)
        input_batches = self.preprocess_input(input_list)
        img_nested_tensor: NestedTensor = input_batches[0]
        images_whwh = img_nested_tensor.tensors.new_tensor(
            [hw[::-1] * 2 for hw in img_nested_tensor.image_sizes]
        )
        targets = self.prepare_targets(input_batches)
        features, output = self.get_det_pred(img_nested_tensor.tensors, images_whwh)
        if self.training:
            match_ids, loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            reid_head_outputs = self.reid_head(features, output, targets, match_ids)
            reid_losses = reid_head_outputs["losses"]
            loss_dict.update(reid_losses)
            output["assign_ids"] = reid_head_outputs["assign_ids"]
            if get_event_storage().iter % self.vis_period == 0:
                self.visualize_training(input_batches, features, output)
            return loss_dict
        else:
            output.pop("anchors", None)
            output.pop("locations", None)
            output.pop("fpn_levels", None)
            logits = output.pop("pred_logits", None)
            bn_scores = logits.sigmoid()  # b x l x 1
            bi_scores = []
            bi_boxes = []
            for bi in range(logits.shape[0]):
                topk_idx = torch.topk(bn_scores[bi].squeeze(1), self.inf_topk)[1]
                bi_scores.append(bn_scores[bi : bi + 1][:, topk_idx])
                bi_boxes.append(output["pred_boxes"][bi : bi + 1][:, topk_idx])
            output["pred_scores"] = torch.cat(bi_scores, dim=0)
            output["pred_boxes"] = torch.cat(bi_boxes, dim=0)
            reid_feats = self.reid_head(
                features, output, targets=None, det_match_indices=None
            )["reid_feats"]
            if self.reid_head.__class__.__name__.find("Roi") == -1:
                bi_feats = []
                for bi in range(logits.shape[0]):
                    topk_idx = torch.topk(bn_scores[bi].squeeze(1), self.inf_topk)[1]
                    bi_feats.append(reid_feats[bi : bi + 1][:, topk_idx])
                reid_feats = torch.cat(bi_feats, dim=0)
            aug_hws = output["pred_boxes"].new_tensor(input_batches[5])
            org_hws = output["pred_boxes"].new_tensor(input_batches[6])  # B x 2
            org_whwh = torch.stack([org_hws[:, 1], org_hws[:, 0]] * 2, dim=1)  # B x 4
            aug_whwh = torch.stack([aug_hws[:, 1], aug_hws[:, 0]] * 2, dim=1)  # B x 4
            sf = org_whwh / aug_whwh
            # back to org abs
            output["pred_boxes"] *= sf.unsqueeze(1)  # B x 1 x 4
            # NOTE feat norm performed in reid head
            if self.cws:
                output["reid_feats"] = reid_feats * output["pred_scores"]
            else:
                output["reid_feats"] = reid_feats
            return output

    def prepare_targets(self, input_batches):
        targets = []
        # NOTE add image tensor for further visualization
        for bi, (fname, imgid, bboxes, ids, aug_hw) in enumerate(
            zip(*input_batches[1:6])
        ):
            targets.append(
                {
                    "file_name": fname,
                    "image_t": (input_batches[0].tensors)[bi],
                    "image_id": imgid,
                    "boxes": bboxes,
                    "ids": ids,
                    "labels": torch.zeros(bboxes.shape[0], dtype=torch.long).to(
                        self.device
                    ),
                    "aug_whwh": bboxes.new_tensor(((aug_hw[1], aug_hw[0]) * 2,)),
                }
            )
        return targets

    def get_det_pred(self, img_batch_tensor, img_whwh):
        # Feature Extraction.
        src = self.backbone(img_batch_tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)
        # Cls & Reg Prediction.
        (
            base_feats,
            outputs_class,
            outputs_coord,
            anchors,
            locations,
            fpn_levels,
        ) = self.det_head(features)
        if self.det_head_type == "center":
            bs, _, h, w = outputs_class.shape
            # We flatten to compute the cost matrices in a batch
            outputs_class = outputs_class.permute(0, 2, 3, 1).reshape(
                bs, h * w, -1
            )  # [batch_size, num_queries, num_classes]
            outputs_coord = outputs_coord.permute(0, 2, 3, 1).reshape(bs, h * w, -1)
            locations = locations.permute(0, 2, 3, 1).reshape(bs, h * w, -1)

        elif self.head_type == "retina":
            bs, _, hw = outputs_class.shape
            # We flatten to compute the cost matrices in a batch
            outputs_class = outputs_class.permute(
                0, 2, 1
            )  # [batch_size, num_queries, num_classes]
            outputs_coord = outputs_coord.permute(0, 2, 1).reshape(bs, hw, -1)
            anchors = anchors.permute(0, 2, 1)

        else:  # 'FCOS'
            bs, _, hw = outputs_class.shape
            # We flatten to compute the cost matrices in a batch
            outputs_class = outputs_class.permute(
                0, 2, 1
            )  # [batch_size, num_queries, num_classes]

            locations = locations.permute(0, 2, 1)

        return base_feats, {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
            "anchors": anchors,
            "locations": locations,
            "fpn_levels": fpn_levels,
        }

    def inf_query(self, input_list):
        if "roi" in self.cfg.REID_HEAD.NAME:
            input_batches = self.preprocess_input([qd["query"] for qd in input_list])
            img_nested_tensor: NestedTensor = input_batches[0]
            src = self.backbone(img_nested_tensor.tensors)
            features = list()
            for f in self.in_features:
                feature = src[f]
                features.append(feature)
            if self.det_head_type == "center":
                features = self.det_head.deconv(features)
                features = [self.det_head.activation(self.det_head.feat1(features))]
            targets = self.prepare_targets(input_batches)
            reid_feats = self.reid_head(
                features, None, targets, det_match_indices=None
            )["reid_feats"]
            for bi, feat in enumerate(reid_feats):
                input_list[bi]["query"]["feat"] = feat.squeeze(0)
            return input_list
        else:
            return input_list

    @torch.no_grad()
    def visualize_training(self, batched_inputs, featmap, batched_dets):
        """
        Args:
            batched_inputs:
                [imgs nested tensor, imgs paths, imgs ids, imgs bboxes, imgs person ids]
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

        threds = [0.15, 0.2, 0.3, 0.5]
        storage = get_event_storage()
        trans_t2img_t = lambda t: t.detach().cpu() * self.pixel_std.cpu().view(
            -1, 1, 1
        ) + self.pixel_mean.cpu().view(-1, 1, 1)
        img_t2rgb = lambda t: (t.permute(1, 2, 0) * 255).numpy()
        samples: NestedTensor = batched_inputs[0]
        annos = []
        padd_h, padd_w = samples.tensors.shape[-2:]
        for fname, imgid, bboxes, ids in zip(*batched_inputs[1:5]):
            annos.append(
                {
                    "file_name": fname,
                    "image_id": imgid,
                    "boxes": bboxes,  # xyxy abs
                    "ids": ids,
                }
            )
        bs = len(annos)
        level_pcas = mlvl_pca_feat(featmap)
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

            visualize_runs = [Visualizer(img_rgb.copy()) for i in range(len(threds))]
            boxes = batched_dets["pred_boxes"][bi].detach().cpu()
            # TODO check if topk needed
            scores = (
                batched_dets["pred_logits"][bi]
                .detach()
                .cpu()
                .sigmoid()
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
