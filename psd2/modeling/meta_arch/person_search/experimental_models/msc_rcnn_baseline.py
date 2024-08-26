from multiprocessing import pool
from turtle import forward
from sklearn.metrics import classification_report
import torch
from ..build import META_ARCH_REGISTRY

from .base import SearchBase
from ...reid_heads import build_reid_head
from psd2.structures.nested_tensor import nested_collate_fn_idvi as nested_collate_fn
from psd2.structures.nested_tensor import NestedTensor
from psd2.utils.events import get_event_storage
from psd2.utils.visualizer import Visualizer, mlvl_pca_feat

import torch.nn as nn
import torch.nn.functional as tF
from psd2.layers.mem_matching_losses import build_loss_layer
import torch.nn.init as init
from typing import Iterable
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    RPNHead,
)

from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import MultiScaleRoIAlign
import psd2.utils.comm as comm
from torchvision.ops import boxes as box_ops

# NOTE single GPU only
@META_ARCH_REGISTRY.register()
class MSC_RCNN(SearchBase):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.in_features = cfg.DETECTOR.MODEL.ROI_HEAD.IN_FEATURES
        det_cfg = cfg.DETECTOR.MODEL
        anchor_cfg = det_cfg.ANCHOR_GENERATOR
        anchor_generator = AnchorGenerator(
            sizes=anchor_cfg.SIZES, aspect_ratios=anchor_cfg.ASPECT_RATIOS
        )
        head = RPNHead(
            in_channels=self.backbone.output_shape()[self.in_features[-1]].channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        rpn_cfg = det_cfg.RPN
        pre_nms_top_n = dict(
            training=rpn_cfg.PRE_NMS_TOPK_TRAIN,
            testing=rpn_cfg.PRE_NMS_TOPK_TEST,
        )
        post_nms_top_n = dict(
            training=rpn_cfg.POST_NMS_TOPK_TRAIN,
            testing=rpn_cfg.POST_NMS_TOPK_TEST,
        )
        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=rpn_cfg.IOU_THRESHOLDS[1],
            bg_iou_thresh=rpn_cfg.IOU_THRESHOLDS[0],
            batch_size_per_image=rpn_cfg.BATCH_SIZE_PER_IMAGE,
            positive_fraction=rpn_cfg.POSITIVE_FRACTION,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=rpn_cfg.NMS_THRESH,
        )
        self.roi_heads = MscRoiHead(
            cfg, self.backbone.output_shape()
        )  # MscRes5RoiHead(cfg, self.backbone.output_shape())
        # build reid head
        head_cfg = cfg.REID_HEAD
        if head_cfg.NAME == "msc_res5":
            self.reid_head = ReidLoss(head_cfg)  # MscReidHead(head_cfg)
        else:
            self.reid_head = build_reid_head(head_cfg)  # Not compatible yet
        lw_cfg = cfg.DETECTOR.LOSS.LOSS_WEIGHTS
        self.lw_rpn_reg = lw_cfg.RPN_BOX_REG
        self.lw_rpn_cls = lw_cfg.RPN_CLS
        self.lw_box_reg = lw_cfg.BOX_REG
        self.lw_box_cls = lw_cfg.CLS

        self.cws = cfg.CWS

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
                    "ids": ids,  # [ -1 ~ max_num ]
                    "labels": ids.long() + 2,  # [ 1 ~ max_num+2 ]
                    "aug_whwh": bboxes.new_tensor(((aug_hw[1], aug_hw[0]) * 2,)),
                }
            )
        return targets

    @torch.no_grad()
    def inf_query(self, input_list):
        input_batches = self.preprocess_input([qd["query"] for qd in input_list])
        images = input_batches[0].to(self.device)
        features = self.backbone(images.tensors)
        feat_list = list()
        for featn in self.in_features:
            featmap = features[featn]
            feat_list.append(featmap)
        targets = self.prepare_targets(input_batches)
        # query
        boxes = [t["boxes"] for t in targets]
        res4_box_features = self.roi_heads.box_roi_pool(
            features, boxes, images.image_sizes
        )

        ms_box_features = self.roi_heads.feat_head(res4_box_features)
        pfeats, _ = self.roi_heads.embedding_head(ms_box_features)

        reid_feats = self.reid_head(pfeats, None, None)["reid_feats"]
        for bi, feat in enumerate(reid_feats):
            input_list[bi]["query"]["feat"] = feat.squeeze(0)
        return input_list

    def forward(self, input_list):
        # TODO distributed loss reweight
        if "query" in input_list[0]:
            return self.inf_query(input_list)
        input_batches = self.preprocess_input(input_list)
        images = input_batches[0].to(self.device)
        targets = self.prepare_targets(input_batches)
        features = self.backbone(images.tensors)
        features = {fk: features[fk] for fk in self.in_features}
        proposals, proposal_losses = self.rpn(images, features, targets)
        output, mid_output, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        if self.training:
            proposal_feats = mid_output["proposal_feats"]
            proposal_pids = mid_output["proposal_pids"]
            proposal_scores = mid_output["propposal_scores"]
            if isinstance(proposal_pids, Iterable):
                proposal_pids = [pids - 2 for pids in proposal_pids]
            else:
                proposal_pids -= 2  # back to [-2 ~ max_num ]

            feat_list = list()
            for featn in self.in_features:
                featmap = features[featn]
                feat_list.append(featmap)
            reid_head_outputs = self.reid_head(
                proposal_feats, proposal_pids, proposal_scores
            )
            reid_losses = reid_head_outputs["losses"]
            assign_ids = output["assign_ids"]
            output["assign_ids"] = [ids - 2 for ids in assign_ids]
            # rename rpn losses to be consistent with detection losses
            proposal_losses["loss_rpn_reg"] = proposal_losses.pop("loss_rpn_box_reg")
            proposal_losses["loss_rpn_cls"] = proposal_losses.pop("loss_objectness")

            # rename losses for tb
            detector_losses["loss_bbox"] = detector_losses.pop("loss_box_reg")
            detector_losses["loss_ce"] = detector_losses.pop("loss_classifier")

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(reid_losses)
            # apply loss weights
            losses["loss_rpn_reg"] *= self.lw_rpn_reg
            losses["loss_rpn_cls"] *= self.lw_rpn_cls
            losses["loss_bbox"] *= self.lw_box_reg
            losses["loss_ce"] *= self.lw_box_cls
            if get_event_storage().iter % self.vis_period == 0:
                self.visualize_training(input_batches, feat_list, output)
            return losses
        else:
            proposals, _ = self.rpn(images, features, None)
            output, mid_output, _ = self.roi_heads(
                features, proposals, images.image_sizes, None
            )
            pfeats = output["pred_embs"]
            reid_feats = self.reid_head(pfeats, None, None)["reid_feats"]
            aug_hws = output["pred_boxes"][0].new_tensor(input_batches[5])
            org_hws = output["pred_boxes"][0].new_tensor(input_batches[6])  # B x 2
            org_whwh = torch.stack([org_hws[:, 1], org_hws[:, 0]] * 2, dim=1)  # B x 4
            aug_whwh = torch.stack([aug_hws[:, 1], aug_hws[:, 0]] * 2, dim=1)  # B x 4
            sf = org_whwh / aug_whwh
            # back to org abs
            output["pred_boxes"] = [
                output["pred_boxes"][bi] * sf[bi].unsqueeze(0)
                for bi in range(len(output["pred_boxes"]))
            ]
            # NOTE feat norm performed in reid head
            if self.cws:
                output["reid_feats"] = reid_feats * output["pred_scores"]
            else:
                output["reid_feats"] = reid_feats
            return output

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


class MscPredictor(nn.Module):
    def __init__(self, cls_in_channels, reg_in_channels, reg_bn=False):
        super().__init__()
        self.cls_score = nn.Linear(cls_in_channels, 2)
        if reg_bn:
            self.bbox_pred = nn.Sequential(
                nn.Linear(reg_in_channels, 2 * 4), nn.BatchNorm1d(2 * 4)
            )
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(reg_in_channels, 2 * 4)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x_cls, x_reg):
        if x_cls.dim() > 2:
            x_cls = torch.flatten(x_cls, start_dim=1)
        if x_reg.dim() > 2:
            x_reg = torch.flatten(x_reg, start_dim=1)
        scores = self.cls_score(x_cls)
        proposal_deltas = self.bbox_pred(x_reg)
        return scores, proposal_deltas


def _inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class SyncRPN(RegionProposalNetwork):
    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        """
        return super().compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        """
        import torchvision.models.detection._utils as det_utils

        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        num_box_samp = torch.tensor(sampled_pos_inds.numel())
        num_cls_samp = torch.tensor(sampled_inds.numel())
        nums = (num_box_samp, num_cls_samp)
        comm.synchronize()
        all_nums = comm.all_gather(nums)
        num_box = sum([nt[0] for nt in all_nums])
        num_box = torch.clamp(num_box / comm.get_world_size(), min=1).item()
        num_cls = sum([nt[1] for nt in all_nums])
        num_cls = torch.clamp(num_cls / comm.get_world_size(), min=1).item()
        box_loss = (
            det_utils.smooth_l1_loss(
                pred_bbox_deltas[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1 / 9,
                size_average=False,
            )
            / num_box
        )

        objectness_loss = (
            tF.binary_cross_entropy_with_logits(
                objectness[sampled_inds], labels[sampled_inds], reduction="none"
            ).sum()
            / num_cls
        )

        return objectness_loss, box_loss


class MscRes5RoiHead(RoIHeads):
    def __init__(self, cfg, input_shape):
        feat_dim = cfg.DETECTOR.MODEL.ROI_HEAD.FEAT_DIM
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        res5_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        box_predictor = MscPredictor(
            feat_dim, res5_out_channels, cfg.DETECTOR.MODEL.ROI_HEAD.BOX_REG_BN
        )
        pooler_cfg = cfg.DETECTOR.MODEL.ROI_HEAD.ROI_POOLER
        pooler_resolution = pooler_cfg.POOLER_RESOLUTION
        sampling_ratio = pooler_cfg.POOLER_SAMPLING_RATIO
        head_cfg = cfg.DETECTOR.MODEL.ROI_HEAD
        pooler = MultiScaleRoIAlign(
            featmap_names=head_cfg.IN_FEATURES,
            output_size=pooler_resolution,
            sampling_ratio=sampling_ratio,
        )
        super().__init__(
            box_roi_pool=pooler,
            box_head=None,
            box_predictor=box_predictor,
            fg_iou_thresh=head_cfg.IOU_THRESHOLDS[1],
            bg_iou_thresh=head_cfg.IOU_THRESHOLDS[0],
            batch_size_per_image=head_cfg.BATCH_SIZE_PER_IMAGE,
            positive_fraction=head_cfg.POSITIVE_FRACTION,
            bbox_reg_weights=None,
            score_thresh=head_cfg.SCORE_THRESH_TEST,
            nms_thresh=head_cfg.NMS_THRESH_TEST,
            detections_per_img=head_cfg.DETECTIONS_PER_IMAGE_TEST,
        )
        self.res5 = _build_res5(
            cfg
        )  # must be "res5" otherwise cannot load resnet pre-trained parameters
        self.in_features = cfg.DETECTOR.MODEL.ROI_HEAD.IN_FEATURES
        scale0_dim = feat_dim // 2
        scale1_dim = feat_dim - scale0_dim
        in_channels = input_shape[self.in_features[0]].channels
        self.scale0_trans = nn.Linear(in_channels, scale0_dim)
        self.scale0_bn = nn.BatchNorm1d(scale0_dim)

        self.scale1_trans = nn.Linear(res5_out_channels, scale1_dim)
        self.scale1_bn = nn.BatchNorm1d(scale1_dim)
        init.normal_(self.scale0_trans.weight, std=0.01)
        init.constant_(self.scale0_trans.bias, 0)
        init.normal_(self.scale1_trans.weight, std=0.01)
        init.constant_(self.scale1_trans.bias, 0)

    def forward(
        self,
        features,
        proposals,
        image_shapes,
        targets=None,
    ):
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert (
                    t["boxes"].dtype in floating_point_types
                ), "target boxes must of float type"
                assert (
                    t["labels"].dtype == torch.int64
                ), "target labels must of int64 type"
                if self.has_keypoint():
                    assert (
                        t["keypoints"].dtype == torch.float32
                    ), "target keypoints must of float type"

        if self.training:
            (
                proposals,
                matched_idxs,
                labels,  # for persons, >0; for bk regions, =0
                regression_targets,
            ) = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features0 = self.box_roi_pool(features, proposals, image_shapes)
        box_features1 = self.res5(box_features0)
        box_embs0 = box_features0.mean(dim=[2, 3])
        box_embs1 = box_features1.mean(dim=[2, 3])
        scale0_embs = self.scale0_trans(box_embs0)  # b x c/2
        scale0_embs_bn = self.scale0_bn(scale0_embs)
        scale1_embs = self.scale1_trans(box_embs1)  # b x c/2
        scale1_embs_bn = self.scale1_bn(scale1_embs)
        class_logits, box_regression = self.box_predictor(
            torch.cat([scale0_embs_bn, scale1_embs_bn], dim=1), box_embs1
        )

        losses = {}
        mid_results = {}
        det_results = {}
        if self.training:
            # un-nms
            assert labels is not None and regression_targets is not None
            det_labels = [lb.clamp(0, 1) for lb in labels]
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, det_labels, regression_targets
            )

            """
            this_num_proposals = torch.tensor(
                sum([dtlb.shape[0] for dtlb in det_labels])
            )
            comm.synchronize()
            # RPN loss are calculated with same num of samples
            all_num_proposals = comm.all_gather(this_num_proposals)
            num_boxes = sum([num for num in all_num_proposals])
            num_boxes = torch.clamp(num_boxes / comm.get_world_size(), min=1).item()
            losses = {
                "loss_classifier": loss_classifier
                * this_num_proposals.item()
                / num_boxes,
                "loss_box_reg": loss_box_reg * this_num_proposals.item() / num_boxes,
            }
            """
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }

            num_splits = [prs.shape[0] for prs in proposals]
            proposal_embs = [
                torch.split(
                    torch.cat([scale0_embs, scale1_embs], dim=1), num_splits, dim=0
                ),
                torch.split(
                    torch.cat([scale0_embs_bn, scale1_embs_bn], dim=1),
                    num_splits,
                    dim=0,
                ),
            ]
            proposal_pids = labels
            proposal_scores = torch.split(
                tF.softmax(class_logits, -1)[:, 1:], num_splits, dim=0
            )
            mid_results = {
                "proposal_feats": proposal_embs,
                "proposal_pids": proposal_pids,
                "propposal_scores": proposal_scores,
            }
            # for vis
            add_lbs = [[pid_img] for pid_img in labels]
            boxes, scores, det_labels, pids = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes, add_lbs
            )
            assign_ids = [pid_img[0] for pid_img in pids]
            det_results["assign_ids"] = assign_ids
        else:
            num_splits = [prs.shape[0] for prs in proposals]
            proposal_embs = [
                torch.split(
                    torch.cat([scale0_embs, scale1_embs], dim=1), num_splits, dim=0
                ),
                torch.split(
                    torch.cat([scale0_embs_bn, scale1_embs_bn], dim=1),
                    num_splits,
                    dim=0,
                ),
            ]
            add_lbs = [[embs0, embs1] for embs0, embs1 in zip(*proposal_embs)]
            # nms out for eval
            boxes, scores, det_labels, pred_embs = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes, add_lbs
            )
            det_results["pred_embs"] = pred_embs
        det_results.update(
            {
                "pred_boxes": boxes,
                "pred_logits": [
                    _inverse_sigmoid(s_img.unsqueeze(1)) for s_img in scores
                ],
            }
        )  # for compatibility

        return det_results, mid_results, losses

    def postprocess_detections(
        self, class_logits, box_regression, proposals, image_shapes, additional_lbs=[]
    ):
        """
        additional_lbs: additional labels per image, len==num_image
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = tF.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_add_labels = []
        for boxes, scores, image_shape, img_addtional_lbs in zip(
            pred_boxes_list, pred_scores_list, image_shapes, additional_lbs
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            img_addtional_lbs = [add_lb[inds] for add_lb in img_addtional_lbs]
            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            img_addtional_lbs = [add_lb[keep] for add_lb in img_addtional_lbs]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            img_addtional_lbs = [add_lb[keep] for add_lb in img_addtional_lbs]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_add_labels.append(img_addtional_lbs)

        return all_boxes, all_scores, all_labels, all_add_labels


class ReidLoss(nn.Module):
    def __init__(self, head_cfg):
        super().__init__()
        pfeats_dim = head_cfg.PERSON_FEATURE.DIM
        loss_cfg = head_cfg.LOSS
        self.loss_layer = build_loss_layer(loss_cfg, head_cfg.PERSON_FEATURE.DIM)

    def forward(self, pfeats, pids, p_scores):
        head_outputs = {}
        if self.training:
            head_outputs["losses"] = {}
            if isinstance(pfeats, Iterable):
                pfeats = torch.cat(pfeats, dim=0)[None]
            if isinstance(pids, Iterable):
                pids = torch.cat(pids, dim=0)[None]
            losses = self.loss_layer(pfeats.flatten(0, -2), pids.view(-1), None)
            head_outputs["assign_ids"] = pids
            # head_outputs["reid_feats"] = reid_feats
            head_outputs["losses"].update(losses)

        else:
            if isinstance(pfeats, Iterable):
                norm_feats = []
                for pfeat in pfeats:
                    norm_feats.append(tF.normalize(pfeat, dim=-1))
            else:
                assert isinstance(pfeats, torch.Tensor)
                norm_feats = tF.normalize(pfeats, dim=-1)
            head_outputs["reid_feats"] = norm_feats
        return head_outputs


class MscRoiHead(RoIHeads):
    def __init__(self, cfg, input_shape):
        feat_dim = cfg.DETECTOR.MODEL.ROI_HEAD.FEAT_DIM
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        res5_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        box_predictor = CoordRegressor(
            res5_out_channels, 2, cfg.DETECTOR.MODEL.ROI_HEAD.BOX_REG_BN
        )
        pooler_cfg = cfg.DETECTOR.MODEL.ROI_HEAD.ROI_POOLER
        pooler_resolution = pooler_cfg.POOLER_RESOLUTION
        sampling_ratio = pooler_cfg.POOLER_SAMPLING_RATIO
        head_cfg = cfg.DETECTOR.MODEL.ROI_HEAD
        pooler = MultiScaleRoIAlign(
            featmap_names=head_cfg.IN_FEATURES,
            output_size=pooler_resolution,
            sampling_ratio=sampling_ratio,
        )
        box_head = Res5BoxHead(cfg)
        super().__init__(
            box_roi_pool=pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=head_cfg.IOU_THRESHOLDS[1],
            bg_iou_thresh=head_cfg.IOU_THRESHOLDS[0],
            batch_size_per_image=head_cfg.BATCH_SIZE_PER_IMAGE,
            positive_fraction=head_cfg.POSITIVE_FRACTION,
            bbox_reg_weights=[10.0, 10.0, 5.0, 5.0],
            score_thresh=head_cfg.SCORE_THRESH_TEST,
            nms_thresh=head_cfg.NMS_THRESH_TEST,
            detections_per_img=head_cfg.DETECTIONS_PER_IMAGE_TEST,
        )
        self.embedding_head = EmbHead(
            featmap_names=["res4", "res5"],
            in_channels=[res5_out_channels // 2, res5_out_channels],
            dim=feat_dim,
        )

    @property
    def feat_head(self):  # re-name
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
            ) = self.select_training_samples(proposals, targets)

        roi_pooled_features = self.box_roi_pool(features, proposals, image_shapes)
        rcnn_features = self.feat_head(roi_pooled_features)
        box_regression = self.box_predictor(rcnn_features["res5"])
        embeddings_, class_logits = self.embedding_head(rcnn_features)

        losses = {}
        mid_results = {}
        det_results = {}
        if self.training:
            det_labels = [y.clamp(0, 1) for y in labels]
            loss_detection, loss_box_reg = det_rcnn_loss(
                class_logits, box_regression, det_labels, regression_targets
            )

            losses = {
                "loss_classifier": loss_detection,
                "loss_box_reg": loss_box_reg,
            }
            num_splits = [prs.shape[0] for prs in proposals]
            proposal_embs = torch.split(embeddings_, num_splits, dim=0)
            proposal_pids = labels
            proposal_scores = torch.split(
                tF.softmax(class_logits, -1)[:, 1:], num_splits, dim=0
            )
            mid_results = {
                "proposal_feats": proposal_embs,
                "proposal_pids": proposal_pids,
                "propposal_scores": proposal_scores,
            }
            with torch.no_grad():
                # for vis
                add_lbs = [[pid_img] for pid_img in labels]
                boxes, scores, det_labels, pids = self.postprocess_detections(
                    class_logits, box_regression, proposals, image_shapes, add_lbs
                )
                assign_ids = [pid_img[0] for pid_img in pids]
                det_results["assign_ids"] = assign_ids
        else:
            num_splits = [prs.shape[0] for prs in proposals]
            proposal_embs = torch.split(embeddings_, num_splits, dim=0)
            add_lbs = [[img_embs] for img_embs in proposal_embs]
            boxes, scores, labels, pred_embs = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes, add_lbs
            )
            embs = [feats[0] for feats in pred_embs]
            det_results["pred_embs"] = embs
        with torch.no_grad():
            scores = [sc.unsqueeze(1) for sc in scores]
            det_results.update(
                {"pred_boxes": boxes, "pred_scores": scores}
            )  # for compatibility
        # Mask and Keypoint losses are deleted
        return det_results, mid_results, losses

    def postprocess_detections(
        self, class_logits, box_regression, proposals, image_shapes, additional_lbs=[]
    ):
        """
        additional_lbs: additional labels per image, len==num_image
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = tF.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_add_labels = []
        for boxes, scores, image_shape, img_addtional_lbs in zip(
            pred_boxes_list, pred_scores_list, image_shapes, additional_lbs
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            img_addtional_lbs = [add_lb[inds] for add_lb in img_addtional_lbs]
            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            img_addtional_lbs = [add_lb[keep] for add_lb in img_addtional_lbs]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            img_addtional_lbs = [add_lb[keep] for add_lb in img_addtional_lbs]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_add_labels.append(img_addtional_lbs)

        return all_boxes, all_scores, all_labels, all_add_labels


class EmbHead(nn.Module):
    def __init__(self, featmap_names=["feat_res5"], in_channels=[2048], dim=256):
        super(EmbHead, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = int(dim)

        self.projectors = nn.ModuleDict()
        self.projectors_reid = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_chennel, indv_dim in zip(
            self.featmap_names, self.in_channels, indv_dims
        ):
            proj = nn.Sequential(
                nn.Linear(in_chennel, int(indv_dim)), nn.BatchNorm1d(int(indv_dim))
            )
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj
        for ftname, in_chennel, indv_dim in zip(
            self.featmap_names, self.in_channels, indv_dims
        ):
            proj = nn.Sequential(
                nn.Linear(in_chennel, int(indv_dim)), nn.BatchNorm1d(int(indv_dim))
            )
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors_reid[ftname] = proj
        self.cls_score = nn.Linear(dim, 2)

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            score = self.cls_score(embeddings)
            emb_reid = self.projectors[k](v)
            return emb_reid, score
            # return embeddings, score
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))
            embeddings = torch.cat(outputs, dim=1)
            score = self.cls_score(embeddings)
            outputs_reid = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs_reid.append(self.projectors_reid[k](v))
            emb_reid = torch.cat(outputs, dim=1)
            return emb_reid, score
            # return embeddings, score

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x  # ndim = 2, (N, d)

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim / parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp


class Res5BoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.res5 = _build_res5(cfg)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, res4_feat):
        res5_feat = self.res5(res4_feat)
        return {"res4": self.gap(res4_feat), "res5": self.gap(res5_feat)}


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
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = tF.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def _build_res5(cfg):
    from psd2.modeling.backbone.resnet import (
        BasicBlock,
        DeformBottleneckBlock,
        BottleneckBlock,
        ResNet,
    )

    norm = cfg.MODEL.RESNETS.NORM
    # fmt: off
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    bottleneck_channels = bottleneck_channels*2**3
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    in_channels         = out_channels*2**2
    out_channels        = out_channels*2**3
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]
    if depth in [18, 34]:
        assert (
            out_channels == 64
        ), "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert (
            res5_dilation == 1
        ), "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"
    idx = 3
    stage_idx = 5
    dilation = res5_dilation if stage_idx == 5 else 1
    first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
    stage_kargs = {
        "num_blocks": num_blocks_per_stage[idx],
        "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "norm": norm,
    }
    # Use BasicBlock for R18 and R34.
    if depth in [18, 34]:
        stage_kargs["block_class"] = BasicBlock
    else:
        stage_kargs["bottleneck_channels"] = bottleneck_channels
        stage_kargs["stride_in_1x1"] = stride_in_1x1
        stage_kargs["dilation"] = dilation
        stage_kargs["num_groups"] = num_groups
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
    blocks = ResNet.make_stage(**stage_kargs)
    return nn.Sequential(*blocks)


def det_rcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Norm-Aware R-CNN.
    Arguments:
        class_logits (Tensor), size = (N, )
        box_regression (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    """classification_loss = tF.binary_cross_entropy_with_logits(
        class_logits, labels.float()
    )"""
    classification_loss = tF.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N = class_logits.size(0)
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = tF.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss
