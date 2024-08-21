import torch
from torch import nn
from ..build import META_ARCH_REGISTRY
from psd2.utils.visualizer import Visualizer, mlvl_pca_feat
from psd2.modeling.matcher import SrcnnHungarianMatcher as Matcher
from psd2.layers.extra_det_head import DynamicHead, DynamicConv, RCNNHead
from psd2.structures.boxes import box_cxcywh_to_xyxy
from psd2.structures.nested_tensor import NestedTensor  # , nested_collate_fn
from psd2.structures.nested_tensor import nested_collate_fn_idvi as nested_collate_fn
from psd2.config import configurable
from .base import SearchBase
from psd2.layers.set_criterion import SrcnnSetCriterion as SetCriterion
from psd2.utils.events import get_event_storage
from psd2.modeling.reid_heads import build_reid_head


@META_ARCH_REGISTRY.register()
class SparsePS_Baseline(SearchBase):
    """
    Implement SparseRCNN for Decoupled Person Search
    """

    @configurable
    def __init__(
        self,
        cfg,
        num_proposals,
        hidden_dim,
        in_features,
        criterion,
        use_focal,
        reid_head,
    ):
        super().__init__(cfg)
        self._local_init(
            cfg, num_proposals, hidden_dim, in_features, criterion, use_focal, reid_head
        )

    def _local_init(
        self,
        cfg,
        num_proposals,
        hidden_dim,
        in_features,
        criterion,
        use_focal,
        reid_head,
    ):
        self.in_features = in_features
        self.num_proposals = num_proposals
        self.hidden_dim = hidden_dim

        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)
        self.criterion = criterion
        self.use_focal = use_focal
        self.deep_supervision = cfg.DETECTOR.LOSS.DEEP_SUPERVISION
        # Build Dynamic Head. TODO move to from config
        # dynamic conv
        pooler_cfg = cfg.MODEL.ROI_BOX_HEAD
        pooler_size = pooler_cfg.POOLER_RESOLUTION
        pooler_type = pooler_cfg.POOLER_TYPE
        pooler_samp_ratio = pooler_cfg.POOLER_SAMPLING_RATIO

        dy_conv_cfg = cfg.DETECTOR.MODEL.DYNAMIC_CONV
        dim_dym = dy_conv_cfg.DIM_DYNAMIC
        num_dym = dy_conv_cfg.NUM_DYNAMIC
        dym_norm = dy_conv_cfg.NORM
        dym_conv = DynamicConv(hidden_dim, dim_dym, num_dym, pooler_size, dym_norm)

        rcnn_head_cfg = cfg.DETECTOR.MODEL.RCNN_HEAD
        num_cls_layers = rcnn_head_cfg.NUM_CLS_LAYERS
        num_reg_layers = rcnn_head_cfg.NUM_REG_LAYERS
        head_norm = rcnn_head_cfg.NORM
        dim_ffd = rcnn_head_cfg.DIM_FEEDFORWARD
        nhead_msa = rcnn_head_cfg.NHEADS_MSA
        head_dropout = rcnn_head_cfg.DROPOUT
        head_activation = rcnn_head_cfg.ACTIVATION
        pfeat_type = rcnn_head_cfg.PFEAT_TYPE
        rcnn_head = RCNNHead(
            hidden_dim,
            1,
            num_cls_layers,
            num_reg_layers,
            use_focal,
            dym_conv,
            head_norm,
            dim_ffd,
            nhead_msa,
            head_dropout,
            head_activation,
            pfeat_type=pfeat_type,
        )
        roi_head_cfg = cfg.MODEL.ROI_HEADS
        self.det_head = DynamicHead(
            pooler_type,
            roi_head_cfg.IN_FEATURES,
            pooler_size,
            pooler_samp_ratio,
            self.backbone.output_shape(),
            rcnn_head,
            rcnn_head_cfg.HEADS_DEPTH,
            self.deep_supervision,
            1,
            use_focal,
            cfg.DETECTOR.LOSS.FOCAL.PRIOR_PROB,
            pfeat_type,
        )

        self.reid_head = reid_head
        # self.det_topk_to_head = cfg.DETECTOR.DET_TOPK_TO_HEAD
        self.cws = cfg.CWS

    @classmethod
    def from_config(cls, cfg):
        detector_cfg = cfg.DETECTOR
        srcnn_cfg = detector_cfg.MODEL
        rcnn_head_cfg = srcnn_cfg.RCNN_HEAD
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        num_proposals = srcnn_cfg.NUM_PROPOSALS
        hidden_dim = rcnn_head_cfg.HIDDEN_DIM
        num_heads = rcnn_head_cfg.HEADS_DEPTH

        # Loss parameters:
        loss_cfg = detector_cfg.LOSS
        loss_weights = loss_cfg.LOSS_WEIGHTS
        class_weight = loss_weights.CLS
        giou_weight = loss_weights.BOX_GIOU
        l1_weight = loss_weights.BOX_L1
        no_object_weight = loss_weights.NO_OBJECT
        deep_supervision = loss_cfg.DEEP_SUPERVISION
        use_focal = loss_cfg.FOCAL.USE_FOCAL
        focal_alpha = loss_cfg.FOCAL.ALPHA
        focal_gamma = loss_cfg.FOCAL.GAMMA

        # Build Criterion.
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
            for i in range(num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes"]

        criterion = SetCriterion(
            num_classes=detector_cfg.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            use_focal=use_focal,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        reid_head = build_reid_head(cfg.REID_HEAD)
        return {
            "cfg": cfg,
            "num_proposals": num_proposals,
            "hidden_dim": hidden_dim,
            "in_features": in_features,
            "criterion": criterion,
            "use_focal": use_focal,
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

    @torch.no_grad()
    def inf_query(self, input_list):
        input_batches = self.preprocess_input([qd["query"] for qd in input_list])
        img_nested_tensor: NestedTensor = input_batches[0]
        # Feature Extraction.
        src = self.backbone(img_nested_tensor.tensors)
        res_feats = src["bk_feats"]
        lateral_feats = list()
        out_feats = src["out"]
        features = list()
        for f in self.in_features:
            feature = out_feats[f]
            features.append(feature)
            lateral_feats.append(src["lateral"][f])
        targets = self.prepare_targets(input_batches)
        if "ms" in self.cfg.REID_HEAD.NAME:
            reid_feats = self.reid_head(
                lateral_feats, None, targets, det_match_indices=None
            )["reid_feats"]
        else:
            reid_feats = self.reid_head(
                features, None, targets, det_match_indices=None
            )["reid_feats"]
        for bi, feat in enumerate(reid_feats):
            input_list[bi]["query"]["feat"] = feat.squeeze(0)
        return input_list

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
        res_feats = src["bk_feats"]
        lateral_feats = list()
        out_feats = src["out"]
        features = list()
        for f in self.in_features:
            feature = out_feats[f]
            features.append(feature)
            lateral_feats.append(src["lateral"][f])

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * img_whwh[:, None, :]  # xyxy_abs in aug

        # Prediction.
        return (
            res_feats,
            lateral_feats,
            features,
            self.det_head(features, proposal_boxes, self.init_proposal_features.weight),
        )

    def forward(self, input_list):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        if "query" in input_list[0]:
            return self.inf_query(input_list)
        input_batches = self.preprocess_input(input_list)
        img_nested_tensor: NestedTensor = input_batches[0]
        images_whwh = img_nested_tensor.tensors.new_tensor(
            [hw[::-1] * 2 for hw in img_nested_tensor.image_sizes]
        )
        targets = self.prepare_targets(input_batches)
        (
            res_feats_dict,
            lateral_feats,
            features,
            (outputs_class, outputs_coord),
        ) = self.get_det_pred(img_nested_tensor.tensors, images_whwh)
        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }  # xyxy boxes in aug

        if self.training:
            # det losses
            if self.deep_supervision:
                output["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]

            match_ids, loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            # reid losses
            if "ms" in self.cfg.REID_HEAD.NAME:
                reid_head_outputs = self.reid_head(
                    lateral_feats, output, targets, match_ids
                )
            else:
                reid_head_outputs = self.reid_head(features, output, targets, match_ids)
            reid_losses = reid_head_outputs["losses"]
            loss_dict.update(reid_losses)
            # for visualization
            # output["reid_feats"] = reid_head_outputs["reid_feats"]
            output["assign_ids"] = reid_head_outputs["assign_ids"]
            if "aux_outputs" in reid_head_outputs:
                for idx, item in enumerate(reid_head_outputs["aux_outputs"]):
                    output["aux_outputs"][idx].update(item)

            if get_event_storage().iter % self.vis_period == 0:
                self.visualize_training(input_batches, features, output)
            return loss_dict

        else:
            if "ms" in self.cfg.REID_HEAD.NAME:
                reid_feats = self.reid_head(
                    lateral_feats, output, targets=None, det_match_indices=None
                )["reid_feats"]
            else:
                reid_feats = self.reid_head(
                    features, output, targets=None, det_match_indices=None
                )["reid_feats"]
            aug_hws = output["pred_boxes"].new_tensor(input_batches[5])
            org_hws = output["pred_boxes"].new_tensor(input_batches[6])  # B x 2
            logits = output.pop("pred_logits", None)
            output["pred_scores"] = logits.sigmoid()
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
