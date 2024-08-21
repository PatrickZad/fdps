#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from cloudpickle.cloudpickle import instance
from numpy import unique
import torch


from torch import nn


from ..build import META_ARCH_REGISTRY


from psd2.structures import Boxes
from psd2.utils.visualizer import pca_feat, Visualizer, mlvl_pca_feat, tsne_embs

from psd2.modeling.matcher import SrcnnHungarianMatcher as Matcher
from psd2.layers.extra_det_head import DynamicHead
from psd2.structures.boxes import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from psd2.structures.nested_tensor import NestedTensor, nested_collate_fn
from psd2.config import configurable
from .base import SearchBase
from psd2.layers.set_criterion import OIMSrcnnSetCriterion as SetCriterion
from torch.nn import init
from psd2.structures import Boxes
from psd2.utils.events import get_event_storage
import psd2.utils.comm as comm


@META_ARCH_REGISTRY.register()
class SparseRCNN_PS_DC(SearchBase):
    """
    Implement SparseRCNN for Decoupled Person Search
    """

    @configurable
    def __init__(
        self, cfg, num_proposals, hidden_dim, in_features, criterion, use_focal
    ):
        super().__init__(cfg)
        self._local_init(
            cfg, num_proposals, hidden_dim, in_features, criterion, use_focal
        )

    def _local_init(
        self, cfg, num_proposals, hidden_dim, in_features, criterion, use_focal
    ):
        self.in_features = in_features
        self.num_proposals = num_proposals
        self.hidden_dim = hidden_dim

        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        self.criterion = criterion
        self.use_focal = use_focal
        # embedding similar to NormAware
        roi_size = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        if isinstance(roi_size, int):
            roi_size = (roi_size, roi_size)
        fpn_out_channels = cfg.MODEL.FPN.OUT_CHANNELS
        self.feat_proj = nn.Linear(
            roi_size[0] * roi_size[1] * fpn_out_channels,
            self.cfg.MODEL.SEARCH.PERSON_FEAT.FEAT_LEN,
        )

        self.feat_bn = nn.BatchNorm1d(self.cfg.MODEL.SEARCH.PERSON_FEAT.FEAT_LEN)

        init.normal_(self.feat_proj.weight, std=0.01)
        init.normal_(self.feat_bn.weight, std=0.01)
        init.constant_(self.feat_proj.bias, 0)
        init.constant_(self.feat_bn.bias, 0)
        self.append_gt = cfg.MODEL.SEARCH.PERSON_FEAT.APPEND_GT
        self.deep_supervision = cfg.MODEL.SEARCH.LOSS_WEIGHTS.DEEP_SUPERVISION
        self.cws_feat = cfg.MODEL.SEARCH.PERSON_FEAT.CWS
        self.reid_box_bp = cfg.MODEL.SEARCH.PERSON_FEAT.BOX_BP
        self.loss_feat_at = cfg.MODEL.SEARCH.PERSON_FEAT.LOSS_FEAT
        self.inf_feat_at = cfg.MODEL.SEARCH.PERSON_FEAT.INF_FEAT

    @classmethod
    def from_config(cls, cfg):
        search_cfg = cfg.MODEL.SEARCH
        srcnn_cfg = search_cfg.SRCNN
        rcnn_head_cfg = srcnn_cfg.RCNN_HEAD
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        num_proposals = srcnn_cfg.NUM_PROPOSALS
        hidden_dim = rcnn_head_cfg.HIDDEN_DIM
        num_heads = rcnn_head_cfg.HEADS_DEPTH

        # Loss parameters:
        loss_cfg = search_cfg.LOSS_WEIGHTS
        class_weight = loss_cfg.CLASS_WEIGHT
        giou_weight = loss_cfg.GIOU_WEIGHT
        l1_weight = loss_cfg.L1_WEIGHT
        no_object_weight = loss_cfg.NO_OBJECT_WEIGHT
        deep_supervision = loss_cfg.DEEP_SUPERVISION
        use_focal = loss_cfg.FOCAL.USE_FOCAL

        # Build Criterion.
        matcher = Matcher(
            cfg=cfg,
            cost_class=class_weight,
            cost_bbox=l1_weight,
            cost_giou=giou_weight,
            use_focal=use_focal,
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
        weight_dict["loss_oim"] = search_cfg.PERSON_FEAT.OIM.LOSS_WEIGHT
        if search_cfg.PERSON_FEAT.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(num_heads - 1):
                aux_weight_dict.update({"loss_oim" + f"_{i}": weight_dict["loss_oim"]})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes"]

        criterion = SetCriterion(
            cfg=cfg,
            num_classes=1,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            use_focal=use_focal,
        )
        return {
            "cfg": cfg,
            "num_proposals": num_proposals,
            "hidden_dim": hidden_dim,
            "in_features": in_features,
            "criterion": criterion,
            "use_focal": use_focal,
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

    @torch.no_grad()
    def visualize_training(
        self, batched_inputs, featmap, batched_dets, gt_feats=None, gt_labels=None
    ):
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
        import random
        import functools

        COLORS = ["r", "g", "b", "y", "c", "m"]
        T_COLORS_BG = {
            "r": "white",
            "g": "white",
            "b": "white",
            "y": "black",
            "c": "black",
            "m": "white",
        }

        def score_split(score, threds):
            for i, v in enumerate(threds[:-1]):
                if score >= threds[i] and score < threds[i + 1]:
                    return i

        threds = [0, 0.05, 0.2, 0.5, 0.7, 1]
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
            ids = annos[bi]["ids"]
            for i in range(boxes.shape[0]):
                visualize_org.draw_box(boxes[i])
                id_pos = boxes[i, :2]
                visualize_org.draw_text(
                    str(ids[i]), id_pos, horizontal_alignment="left", color="w"
                )

            rgb_org_ann = visualize_org.get_output().get_image()
            t_org_ann = tvF.to_tensor(rgb_org_ann)
            storage.put_image("img_{}/gt".format(bi), t_org_ann)

            visualize_runs = [
                Visualizer(img_rgb.copy()) for i in range(len(threds) - 1)
            ]
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
                split = score_split(score, threds)
                b_clr = COLORS[i % len(COLORS)]
                t_clr = T_COLORS_BG[b_clr]
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
        """pfeats = batched_dets["reid_feats"]
        pids = batched_dets["assign_ids"]
        feat_id = [pfeats, pids]
        if gt_feats is not None and gt_labels is not None:
            feat_id += [gt_feats, gt_labels]
        comm.synchronize()
        all_feat_ids = comm.gather(feat_id, dst=0)
        if comm.is_main_process():
            fl_feats = []
            fl_ids = []
            if gt_feats is not None and gt_labels is not None:
                for feats, ids, gtfeats, gtlabels in all_feat_ids:
                    fl_feats.append(feats.cpu().flatten(0, 1))
                    fl_ids.append(ids.cpu().reshape(-1))
                    for b_gtfeats, b_gtids in zip(gtfeats, gtlabels):
                        fl_feats.append(b_gtfeats.cpu())
                        fl_ids.append(b_gtids.cpu())
            else:
                for feats, ids in all_feat_ids:
                    fl_feats.append(feats.cpu().flatten(0, 1))
                    fl_ids.append(ids.cpu().reshape(-1))
            pfeats, pids = torch.cat(fl_feats, dim=0), torch.cat(fl_ids, dim=0)
            mask = pids > -1
            if pfeats[mask].shape[0] < 1:
                return
            uniq_pids = pids[mask].reshape(-1).tolist()
            if len(uniq_pids) > 10:
                vis_ids = random.sample(uniq_pids, 10)
                mask = functools.reduce(
                    torch.logical_or, [pids == uid for uid in vis_ids]
                )
            tsne_rgb = tsne_embs(pfeats[mask], pids[mask])
            storage.put_image("embs", tvF.to_tensor(tsne_rgb))"""

    def pfeat_head(self, roi_feats):
        proj_feats = self.feat_proj(roi_feats)
        if self.training:
            feat_at = self.loss_feat_at
        else:
            feat_at = self.inf_feat_at
        if feat_at == "after_bn":
            return self.feat_bn(proj_feats)
        if feat_at == "before_bn":
            return proj_feats

    def inf_query(self, input_list):
        input_batches = self.preprocess_input([qd["query"] for qd in input_list])
        img_nested_tensor: NestedTensor = input_batches[0]
        # Feature Extraction.
        src = self.backbone(img_nested_tensor.tensors)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)
        box_pooler = self.head.box_pooler
        boxes_list = []
        for bboxes in input_batches[3]:
            boxes_list.append(Boxes(bboxes))
        gt_rois_feats = box_pooler(features, boxes_list)  # B x 256 x 7 x 7
        gt_psfeats = self.pfeat_head(gt_rois_feats.flatten(start_dim=1))  # B x 256
        for bi, feat in enumerate(gt_psfeats):
            input_list[bi]["query"]["feat"] = feat.squeeze(0)
        return input_list

    def prepare_targets(self, input_batches):
        targets = []
        for fname, imgid, bboxes, ids, aug_hw in zip(*input_batches[1:6]):
            targets.append(
                {
                    "file_name": fname,
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

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * img_whwh[:, None, :]  # xyxy_abs in aug

        # Prediction.
        return features, self.head(
            features, proposal_boxes, self.init_proposal_features.weight
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
        features, (outputs_class, outputs_coord) = self.get_det_pred(
            img_nested_tensor.tensors, images_whwh
        )

        # scale xyxy abs coords to [0,1] in aug_size
        aug_hw_bs = images_whwh.new_tensor(input_batches[5])  # B x 2
        aug_whwh_bs = torch.stack(
            [aug_hw_bs[:, 1], aug_hw_bs[:, 0]] * 2, dim=1
        )  # B x 4
        box_pooler = self.head.box_pooler
        nd, nb, nq, lb = outputs_coord.shape

        roi_boxes = outputs_coord.permute(1, 0, 2, 3).reshape(
            nb, -1, lb
        )  # B x (D x Nq) x 4

        if self.training and self.append_gt:
            # search loss makes no difference to learned boxes
            if self.reid_box_bp:
                dt_roi_boxes = roi_boxes
            else:
                dt_roi_boxes = roi_boxes.detach()
            boxes_list = [
                Boxes(torch.cat([dt_roi_boxes[bi], input_batches[3][bi]]))
                for bi in range(nb)
            ]  # append gt boxes
            rois_feats = box_pooler(features, boxes_list)  # (Bx[DxNq+g]) x 256 x 7 x 7
            rois_feats = rois_feats.flatten(start_dim=1)
            psfeats = self.pfeat_head(rois_feats)  # (Bx[DxNq+g]) x 256
            num_gts = [input_batches[3][bi].shape[0] for bi in range(nb)]
            num_qs = [nd * nq for _ in range(nb)]
            split_sizes = []
            for bi in range(nb):
                split_sizes.append(num_qs[bi])
                split_sizes.append(num_gts[bi])
            psfeats_splits = torch.split(psfeats, split_sizes)
            pred_psfeats = (
                torch.cat(psfeats_splits[::2], dim=0)
                .view(nb, nd, nq, -1)
                .permute(1, 0, 2, 3)
            )  # D x B x Nq x 256
            gt_psfeats = torch.cat(psfeats_splits[1::2], dim=0)
            ids_list = []
            for ids in zip(input_batches[4]):
                ids_list.append(
                    torch.tensor(ids, dtype=torch.int, device=self.device).squeeze(0)
                )
            gt_ids = torch.cat(ids_list, dim=-1).view(-1)
        else:
            if self.reid_box_bp:
                dt_roi_boxes = roi_boxes
            else:
                dt_roi_boxes = roi_boxes.detach()
            boxes_list = [
                Boxes(dt_roi_boxes[bi]) for bi in range(nb)
            ]  # append gt boxes
            rois_feats = box_pooler(features, boxes_list)  # (BxDxNq) x 256 x 7 x 7
            rois_feats = rois_feats.flatten(start_dim=1)
            pred_psfeats = self.pfeat_head(rois_feats)  # (BxDxNq) x 256
            pred_psfeats = pred_psfeats.view(nb, nd, nq, -1).permute(
                1, 0, 2, 3
            )  # D x B x Nq x 256

        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "reid_feats": pred_psfeats[-1],
        }  # xyxy boxes in aug [0,1]

        if self.training:
            if self.deep_supervision:
                output["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b, "reid_feats": c}
                    for a, b, c in zip(
                        outputs_class[:-1], outputs_coord[:-1], pred_psfeats[:-1]
                    )
                ]
            if self.append_gt:
                loss_dict = self.criterion(
                    output,
                    targets,
                    gt_feats_ids={"reid_feats": gt_psfeats, "ids": gt_ids},
                )
            else:
                loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            if get_event_storage().iter % self.vis_period == 0:
                self.visualize_training(
                    input_batches,
                    features,
                    output,
                    list(gt_psfeats.split(num_gts)),
                    ids_list,
                )
            return loss_dict

        else:
            aug_hws = output["pred_boxes"].new_tensor(input_batches[5])
            org_hws = output["pred_boxes"].new_tensor(input_batches[6])  # B x 2
            logits = output.pop("pred_logits", None)
            output["pred_scores"] = logits.sigmoid()
            org_whwh = torch.stack([org_hws[:, 1], org_hws[:, 0]] * 2, dim=1)  # B x 4
            aug_whwh = torch.stack([aug_hws[:, 1], aug_hws[:, 0]] * 2, dim=1)  # B x 4
            sf = org_whwh / aug_whwh
            # back to org abs
            output["pred_boxes"] *= sf.unsqueeze(1)  # B x 1 x 4
            if self.cws_feat:
                output["reid_feats"] *= output["pred_scores"]
            inter_outs = output.pop("aux_outputs", None)
            return output
