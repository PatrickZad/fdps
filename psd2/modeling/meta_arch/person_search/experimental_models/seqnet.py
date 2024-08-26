################Resnet part#####################
from collections import OrderedDict
import enum
from numpy import dtype, true_divide
from torch.functional import Tensor

import torch.nn.functional as F
import torchvision
from torch import nn

from psd2.structures.boxes import box_xyxy_to_cxcywh


class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(Backbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])


class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


def build_resnet(name="resnet50", pretrained=True):
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)

    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet), Res5Head(resnet)


################OIM part########################
import torch
import torch.nn.functional as F
from torch import autograd, nn


class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets, lut, cq, header, momentum)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, lut, cq, header, momentum = ctx.saved_tensors

        # inputs, targets = tensor_gather((inputs, targets))

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y in zip(inputs, targets):
            if y < len(lut):
                lut[y] = momentum * lut[y] + (1.0 - momentum) * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM.apply(
        inputs, targets, lut, cq, torch.tensor(header), torch.tensor(momentum)
    )


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))

        self.header_cq = 0

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        inds = label >= 0
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        projected = oim(
            inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum
        )
        projected *= self.oim_scalar

        self.header_cq = (
            self.header_cq + (label >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled
        lb_mask = label < 5554
        if label[lb_mask].shape[0] == 0:
            loss_oim = projected.new_tensor(0)
        else:
            loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
        return loss_oim


################Person search part#############
from copy import deepcopy

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

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class SeqNet(nn.Module):
    def __init__(self, cfg):
        super(SeqNet, self).__init__()
        rpn_cfg = cfg.MODEL.SEARCH.SEQNET.RPN
        roi_head_cfg = cfg.MODEL.SEARCH.SEQNET.ROI_HEAD
        oim_cfg = cfg.MODEL.SEARCH.OIM
        backbone, box_head = build_resnet(name="resnet50", pretrained=True)

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=rpn_cfg.PRE_NMS_TOPN_TRAIN,
            testing=rpn_cfg.PRE_NMS_TOPN_TEST,
        )
        post_nms_top_n = dict(
            training=rpn_cfg.POST_NMS_TOPN_TRAIN,
            testing=rpn_cfg.POST_NMS_TOPN_TEST,
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=rpn_cfg.POS_THRESH_TRAIN,
            bg_iou_thresh=rpn_cfg.NEG_THRESH_TRAIN,
            batch_size_per_image=rpn_cfg.BATCH_SIZE_TRAIN,
            positive_fraction=rpn_cfg.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=rpn_cfg.NMS_THRESH,
        )

        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        reid_head = deepcopy(box_head)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        box_predictor = BBoxRegressor(2048, num_classes=2, bn_neck=roi_head_cfg.BN_NECK)

        roi_heads = SeqRoIHeads(
            # OIM
            num_pids=oim_cfg.LUT_SIZE,
            num_cq_size=oim_cfg.CQ_SIZE,
            oim_momentum=oim_cfg.OIM_MOMENTUM,
            oim_scalar=oim_cfg.OIM_SCALAR,
            # SeqNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=roi_head_cfg.POS_THRESH_TRAIN,
            bg_iou_thresh=roi_head_cfg.NEG_THRESH_TRAIN,
            batch_size_per_image=roi_head_cfg.BATCH_SIZE_TRAIN,
            positive_fraction=roi_head_cfg.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=roi_head_cfg.SCORE_THRESH_TEST,
            nms_thresh=roi_head_cfg.NMS_THRESH_TEST,
            detections_per_img=roi_head_cfg.DETECTIONS_PER_IMAGE_TEST,
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
            )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        lw_cfg = cfg.MODEL.SEARCH.LOSS_WEIGHTS
        self.lw_rpn_reg = lw_cfg.LW_RPN_REG
        self.lw_rpn_cls = lw_cfg.LW_RPN_CLS
        self.lw_proposal_reg = lw_cfg.LW_PROPOSAL_REG
        self.lw_proposal_cls = lw_cfg.LW_PROPOSAL_CLS
        self.lw_box_reg = lw_cfg.LW_BOX_REG
        self.lw_box_cls = lw_cfg.LW_BOX_CLS
        self.lw_box_reid = lw_cfg.LW_BOX_REID
        trained_model_path = cfg.MODEL.SEARCH.SEQNET.WEIGHTS
        self._resume_from_ckpt(trained_model_path)
        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )

    @property
    def device(self):
        return self.pixel_mean.device

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
            """
            result_list = input_list.copy()
            
            images = []
            org_hws = []
            for anno_qg in result_list:
                anno = anno_qg["query"]
                img_t = tvF.to_tensor(Image.open(anno["file_name"])).to(self.device)
                images.append(img_t)
                tgt = {
                    "file_name": anno["file_name"],
                    "image_id": anno["image_id"],
                    "boxes": img_t.new_tensor([anno["annotations"][0]["bbox"]]),
                    "pid": [anno["annotations"][0]["person_id"]],
                }
                targets.append(tgt)
                org_hws.append(img_t.shape[-2:])
            images, targets = self.transform(images, targets)
            """
            features = self.backbone(images.tensors)
            # query
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(
                features, boxes, images.image_sizes
            )
            box_features = self.roi_heads.reid_head(box_features)
            embeddings, _ = self.roi_heads.embedding_head(box_features)
            p_feats = embeddings.split(1, 0)[0].cpu()
            result_list = input_list.copy()
            for bi, feat in enumerate(p_feats):
                result_list[bi]["query"]["feat"] = feat
            return result_list
        else:
            input_batches = self.preproxess_input(input_list)
            images = input_batches[0]
            targets = None
            org_hws = input_batches[6]
            features = self.backbone(images.tensors)
            proposals, _ = self.rpn(images, features, targets)
            detections, _ = self.roi_heads(
                features, proposals, images.image_sizes, targets, False
            )
            detections_post = self.transform.postprocess(
                detections, images.image_sizes, org_hws
            )
            return_result = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for det in detections_post:
                """return_result["pred_boxes"].append(
                    box_xyxy_to_cxcywh(det["boxes"].view(-1, 4))
                )"""
                return_result["pred_boxes"].append(det["boxes"].view(-1, 4))
                return_result["pred_scores"].append(det["scores"].view(-1, 1))
                return_result["reid_feats"].append(det["embeddings"].view(-1, 256))
            # vis_inf(input_batches, return_result, "outputs/test_debug", 0.0)
            return return_result

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

        # images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        _, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        # rename rpn losses to be consistent with detection losses
        proposal_losses["loss_rpn_reg"] = proposal_losses.pop("loss_rpn_box_reg")
        proposal_losses["loss_rpn_cls"] = proposal_losses.pop("loss_objectness")

        # rename losses for tb
        detector_losses["loss_bbox_0"] = detector_losses.pop("loss_proposal_reg")
        detector_losses["loss_ce_0"] = detector_losses.pop("loss_proposal_cls")
        detector_losses["loss_bbox"] = detector_losses.pop("loss_box_reg")
        detector_losses["loss_ce"] = detector_losses.pop("loss_box_cls")
        detector_losses["loss_oim"] = detector_losses.pop("loss_box_reid")

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_bbox_0"] *= self.lw_proposal_reg
        losses["loss_ce_0"] *= self.lw_proposal_cls
        losses["loss_bbox"] *= self.lw_box_reg
        losses["loss_ce"] *= self.lw_box_cls
        losses["loss_oim"] *= self.lw_box_reid
        return losses

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


def vis_inf(
    batched_inputs,
    batched_dets,
    out_dir,
    thred=0.5,
):
    """
    Args:
        batched_inputs:
            [imgs nested tensor, imgs paths, imgs ids, imgs bboxes, imgs person ids,
            img_aug_hws, img_org_hws, img_org_boxes,]
        batched_dets:
            {
                "pred_scores": batched person scores B x N x 1,
                "pred_boxes": batched xyxy bboxes in org,
                "pred_feats": batch reid feats
            }
        thred: cls score threshold
    """
    from psd2.structures.boxes import box_cxcywh_to_xyxy
    from psd2.utils.visualizer import Visualizer
    import os

    COLORS = ["r", "g", "b", "y", "c", "m"]
    T_COLORS_BG = {
        "r": "white",
        "g": "white",
        "b": "white",
        "y": "black",
        "c": "black",
        "m": "white",
    }
    bs = len(batched_dets["pred_scores"])
    for bi in range(bs):
        img_path = batched_inputs[1][bi]
        img_id = batched_inputs[2][bi]
        img = Image.open(img_path)
        vis_org = Visualizer(img.copy())
        vis_det = Visualizer(img.copy())
        org_boxes = batched_inputs[7][bi]
        org_ids = batched_inputs[4][bi]
        det_boxes = box_cxcywh_to_xyxy(batched_dets["pred_boxes"][bi])
        det_boxes = det_boxes.cpu().numpy()
        det_scores = batched_dets["pred_scores"][bi].squeeze(1).cpu().numpy().tolist()
        for i in range(org_boxes.shape[0]):
            vis_org.draw_box(org_boxes[i].numpy())
            id_pos = org_boxes[i, :2]
            vis_org.draw_text(
                str(org_ids[i]), id_pos, horizontal_alignment="left", color="w"
            )
        save_dir = os.path.join(out_dir, "inf_vis")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        vis_org.get_output().save(os.path.join(save_dir, img_id[:-4] + "_org.jpg"))
        for i, score in enumerate(det_scores):
            if score >= thred:
                b_clr = COLORS[i % len(COLORS)]
                t_clr = T_COLORS_BG[b_clr]
                vis_det.draw_box(det_boxes[i], edge_color=b_clr)
                vis_det.draw_text(
                    "%.2f" % score,
                    det_boxes[i][2:],
                    horizontal_alignment="right",
                    color=t_clr,
                    bg_color=b_clr,
                )
        vis_det.get_output().save(os.path.join(save_dir, img_id[:-4] + "_det.jpg"))


class SeqRoIHeads(RoIHeads):
    def __init__(
        self,
        num_pids,
        num_cq_size,
        oim_momentum,
        oim_scalar,
        faster_rcnn_predictor,
        reid_head,
        *args,
        **kwargs,
    ):
        super(SeqRoIHeads, self).__init__(*args, **kwargs)
        self.embedding_head = NormAwareEmbedding()
        self.reid_loss = OIMLoss(256, num_pids, num_cq_size, oim_momentum, oim_scalar)
        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.reid_head = reid_head
        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections

    def forward(
        self,
        features,
        proposals,
        image_shapes,
        targets=None,
        query_img_as_gallery=False,
    ):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if self.training:
            (
                proposals,
                _,
                proposal_pid_labels,
                proposal_reg_targets,
            ) = self.select_training_samples(proposals, targets)

        # ------------------- Faster R-CNN head ------------------ #
        proposal_features = self.box_roi_pool(features, proposals, image_shapes)
        proposal_features = self.box_head(proposal_features)
        proposal_cls_scores, proposal_regs = self.faster_rcnn_predictor(
            proposal_features["feat_res5"]
        )

        if self.training:
            boxes = self.get_boxes(proposal_regs, proposals, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            boxes, _, box_pid_labels, box_reg_targets = self.select_training_samples(
                boxes, targets
            )
        else:
            # invoke the postprocess method inherited from parent class to process proposals
            boxes, scores, _ = self.postprocess_proposals(
                proposal_cls_scores, proposal_regs, proposals, image_shapes
            )

        cws = True
        gt_det = None
        if not self.training and query_img_as_gallery:
            # When regarding the query image as gallery, GT boxes may be excluded
            # from detected boxes. To avoid this, we compulsorily include GT in the
            # detection results. Additionally, CWS should be disabled as the
            # confidences of these people in query image are 1
            cws = False
            gt_box = [targets[0]["boxes"]]
            gt_box_features = self.box_roi_pool(features, gt_box, image_shapes)
            gt_box_features = self.reid_head(gt_box_features)
            embeddings, _ = self.embedding_head(gt_box_features)
            gt_det = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

        # no detection predicted by Faster R-CNN head in test phase
        # NOTE not compatible with multi-image inference
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes = gt_det["boxes"] if gt_det else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            embeddings = gt_det["embeddings"] if gt_det else torch.zeros(0, 256)
            return [
                dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)
            ], []

        # --------------------- Baseline head -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features = self.reid_head(box_features)
        box_regs = self.box_predictor(box_features["feat_res5"])
        box_embeddings, box_cls_scores = self.embedding_head(box_features)
        if box_cls_scores.dim() == 0:
            box_cls_scores = box_cls_scores.unsqueeze(0)

        result, losses = [], {}
        if self.training:
            proposal_labels = [y.clamp(0, 1) for y in proposal_pid_labels]
            box_labels = [y.clamp(0, 1) for y in box_pid_labels]
            losses = detection_losses(
                proposal_cls_scores,
                proposal_regs,
                proposal_labels,
                proposal_reg_targets,
                box_cls_scores,
                box_regs,
                box_labels,
                box_reg_targets,
            )
            loss_box_reid = self.reid_loss(box_embeddings, box_pid_labels)
            losses.update(loss_box_reid=loss_box_reid)
        else:
            # The IoUs of these boxes are higher than that of proposals,
            # so a higher NMS threshold is needed
            orig_thresh = self.nms_thresh
            self.nms_thresh = 0.5
            boxes, scores, embeddings, labels = self.postprocess_boxes(
                box_cls_scores,
                box_regs,
                box_embeddings,
                boxes,
                image_shapes,
                fcs=scores,
                gt_det=gt_det,
                cws=cws,
            )
            # set to original thresh after finishing postprocess
            self.nms_thresh = orig_thresh
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
        return result, losses

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
            pred_scores = fcs[0]  # torch.cat(fcs, dim=0)
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
            k, v = featmaps.items()[0]
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
    proposal_cls_scores,
    proposal_regs,
    proposal_labels,
    proposal_reg_targets,
    box_cls_scores,
    box_regs,
    box_labels,
    box_reg_targets,
):
    proposal_labels = torch.cat(proposal_labels, dim=0)
    box_labels = torch.cat(box_labels, dim=0)
    proposal_reg_targets = torch.cat(proposal_reg_targets, dim=0)
    box_reg_targets = torch.cat(box_reg_targets, dim=0)

    loss_proposal_cls = F.cross_entropy(proposal_cls_scores, proposal_labels)
    loss_box_cls = F.binary_cross_entropy_with_logits(
        box_cls_scores, box_labels.float()
    )

    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(proposal_labels > 0).squeeze(1)
    labels_pos = proposal_labels[sampled_pos_inds_subset]
    N = proposal_cls_scores.size(0)
    proposal_regs = proposal_regs.reshape(N, -1, 4)

    loss_proposal_reg = F.smooth_l1_loss(
        proposal_regs[sampled_pos_inds_subset, labels_pos],
        proposal_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_proposal_reg = loss_proposal_reg / proposal_labels.numel()

    sampled_pos_inds_subset = torch.nonzero(box_labels > 0).squeeze(1)
    labels_pos = box_labels[sampled_pos_inds_subset]
    N = box_cls_scores.size(0)
    box_regs = box_regs.reshape(N, -1, 4)

    loss_box_reg = F.smooth_l1_loss(
        box_regs[sampled_pos_inds_subset, labels_pos],
        box_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_box_reg = loss_box_reg / box_labels.numel()

    return dict(
        loss_proposal_cls=loss_proposal_cls,
        loss_proposal_reg=loss_proposal_reg,
        loss_box_cls=loss_box_cls,
        loss_box_reg=loss_box_reg,
    )
