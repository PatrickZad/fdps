import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIPool, nms

# from datasets.data_processing import img_preprocessing

from psd2.org_model_lib.oim_base.models.backbone import Backbone
from psd2.org_model_lib.oim_base.models.head import Head
from psd2.org_model_lib.oim_base.oim.labeled_matching_layer import LabeledMatchingLayer
from psd2.org_model_lib.oim_base.oim.unlabeled_matching_layer import (
    UnlabeledMatchingLayer,
)
from psd2.org_model_lib.oim_base.rpn.proposal_target_layer import ProposalTargetLayer
from psd2.org_model_lib.oim_base.rpn.rpn_layer import RPN
from psd2.org_model_lib.oim_base.utils.boxes import bbox_transform_inv, clip_boxes
from psd2.org_model_lib.oim_base.utils.config import cfg
from psd2.org_model_lib.oim_base.utils.utils import smooth_l1_loss
from ..build import META_ARCH_REGISTRY
import os
import logging
from psd2.structures.nested_tensor import nested_collate_fn

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class OIM_Base(nn.Module):
    """
    Person search network.

    Paper: Joint Detection and Identification Feature Learning for Person Search
           Tong Xiao, Shuang Li, Bochao Wang, Liang Lin, Xiaogang Wang
    """

    def __init__(self, d2_cfg):
        super(OIM_Base, self).__init__()
        rpn_depth = 1024  # Depth of the feature map fed into RPN
        num_classes = 2  # Background and foreground
        self.backbone = Backbone()
        self.head = Head()
        self.rpn = RPN(rpn_depth)
        self.roi_pool = RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.cls_score = nn.Linear(2048, num_classes)
        self.bbox_pred = nn.Linear(2048, num_classes * 4)
        self.feature = nn.Linear(2048, 256)
        self.proposal_target_layer = ProposalTargetLayer(num_classes)
        self.labeled_matching_layer = LabeledMatchingLayer()
        self.unlabeled_matching_layer = UnlabeledMatchingLayer()

        self.freeze_blocks()

        trained_model_path = d2_cfg.MODEL.SEARCH.OIM_BASE.CHKP
        trained_backbone_path = d2_cfg.MODEL.SEARCH.OIM_BASE.BKBN
        if len(trained_model_path) > 0:
            self._resume_from_ckpt(trained_model_path)
        elif len(trained_backbone_path) > 0:
            state_dict = torch.load(trained_backbone_path)
            self.load_state_dict(
                {k: v for k, v in state_dict.items() if k in self.state_dict()}
            )
            logger.info("Loaded pretrained model from: %s" % trained_backbone_path)
        self.register_buffer(
            "pixel_mean", torch.Tensor(d2_cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(d2_cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )

    @property
    def device(self):
        return self.pixel_mean.device

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

    def forward(self, input_list):
        """
        Args:
            img (Tensor): Single image data.
            img_info (Tensor): (height, width, scale)
            gt_boxes (Tensor): Ground-truth boxes in (x1, y1, x2, y2, class, person_id) format.
            probe_roi (Tensor): Take probe_roi as proposal instead of using RPN.

        Returns:
            proposals (Tensor): Region proposals produced by RPN in (0, x1, y1, x2, y2) format.
            probs (Tensor): Classification probability of these proposals.
            proposal_deltas (Tensor): Proposal regression deltas.
            features (Tensor): Extracted features of these proposals.
            rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox and loss_oim (Tensor): Training losses.
        """
        if not self.training:
            return self.inference(input_list)

        input_batches = self.preprocess_input(input_list)
        images_nested = input_batches[0].to(self.device)
        img = images_nested.tensors
        aug_hws = img.new_tensor(input_batches[5])
        org_hws = img.new_tensor(input_batches[6])
        scale = aug_hws[:, :1] / org_hws[:, :1]
        img_info = torch.cat([aug_hws, scale], dim=-1).squeeze(0)
        pids = torch.tensor(input_batches[4], dtype=torch.float32, device=self.device)
        gt_boxes = torch.cat(
            [
                input_batches[3][0][None],
                torch.full_like(pids, 1)[..., None],
                pids[..., None],
            ],
            dim=-1,
        ).squeeze(0)
        probe_roi = None
        assert img.size(0) == 1, "Single batch only."
        # Extract basic feature from image data
        base_feat = self.backbone(images_nested.tensors)

        if probe_roi is None:
            # Feed basic feature map to RPN to obtain rois
            proposals, rpn_loss_cls, rpn_loss_bbox = self.rpn(
                base_feat, img_info, gt_boxes
            )
        else:
            # Take given probe_roi as proposal if probe_roi is not None
            proposals, rpn_loss_cls, rpn_loss_bbox = probe_roi, 0, 0

        # Sample some proposals and assign them ground-truth targets
        (
            proposals,
            cls_labels,
            pid_labels,
            gt_proposal_deltas,
            proposal_inside_ws,
            proposal_outside_ws,
        ) = self.proposal_target_layer(proposals, gt_boxes)

        # RoI pooling based on region proposals
        pooled_feat = self.roi_pool(base_feat, proposals)

        # Extract the features of proposals
        proposal_feat = self.head(pooled_feat).squeeze(2).squeeze(2)

        scores = self.cls_score(proposal_feat)
        probs = F.softmax(scores, dim=1)
        proposal_deltas = self.bbox_pred(proposal_feat)
        features = F.normalize(self.feature(proposal_feat))

        loss_cls = F.cross_entropy(scores, cls_labels)
        loss_bbox = smooth_l1_loss(
            proposal_deltas,
            gt_proposal_deltas,
            proposal_inside_ws,
            proposal_outside_ws,
        )

        # OIM loss
        labeled_matching_scores = self.labeled_matching_layer(features, pid_labels)
        labeled_matching_scores *= 10
        unlabeled_matching_scores = self.unlabeled_matching_layer(features, pid_labels)
        unlabeled_matching_scores *= 10
        matching_scores = torch.cat(
            (labeled_matching_scores, unlabeled_matching_scores), dim=1
        )
        pid_labels = pid_labels.clone()
        pid_labels[pid_labels == -2] = -1
        loss_oim = F.cross_entropy(matching_scores, pid_labels, ignore_index=-1)

        losses = {}

        # apply loss weights
        losses["loss_rpn_reg"] = rpn_loss_bbox
        losses["loss_rpn_cls"] = rpn_loss_cls
        losses["loss_bbox"] = loss_bbox
        losses["loss_ce"] = loss_cls
        losses["loss_oim"] = loss_oim
        return losses

    def freeze_blocks(self):
        """
        The reason why we freeze all BNs in the backbone: The batch size is 1
        in the backbone, so BN is not stable.

        Reference: https://github.com/ShuangLI59/person_search/issues/87
        """
        for p in self.backbone.SpatialConvolution_0.parameters():
            p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                for p in m.parameters():
                    p.requires_grad = False

        # Frozen all bn layers in backbone
        self.backbone.apply(set_bn_fix)

    def train(self, mode=True):
        """
        It's not enough to just freeze all BNs in backbone.
        Setting them to eval mode is also needed.
        """
        nn.Module.train(self, mode)

        if mode:
            # Set all bn layers in backbone to eval mode
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.eval()

            self.backbone.apply(set_bn_eval)

    def inference(self, input_list, threshold=0.75):
        """
        End to end inference. Specific behavior depends on probe_roi.
        If probe_roi is None, detect persons in the image and extract their features.
        Otherwise, extract the feature of the probe RoI in the image.

        Args:
            img (np.ndarray[H, W, C]): Image of BGR order.
            probe_roi (np.ndarray[4]): The RoI to be extracting feature.
            threshold (float): The threshold used to remove those bounding boxes with low scores.

        Returns:
            detections (Tensor[N, 5]): Detected person bounding boxes in
                                       (x1, y1, x2, y2, score) format.
            features (Tensor[N, 256]): Features of these bounding boxes.
        """
        if "query" in input_list[0]:
            input_batches = self.preprocess_input([qd["query"] for qd in input_list])
            probe_roi = input_batches[3]
        else:
            input_batches = self.preprocess_input(input_list)
            probe_roi = None
        images_nested = input_batches[0].to(self.device)
        processed_img = images_nested.tensors
        aug_hws = processed_img.new_tensor(input_batches[5])
        org_hws = processed_img.new_tensor(input_batches[6])
        scale = aug_hws[0][:1] / org_hws[0][:1]
        img_info = torch.cat([aug_hws[0], scale], dim=-1)
        base_feat = self.backbone(images_nested.tensors)
        aug_hws = aug_hws[0]
        org_hws = org_hws[0]
        if "query" in input_list[0]:
            # RoI pooling based on region proposals
            pooled_feat = self.roi_pool(base_feat, probe_roi)
            # Extract the features of proposals
            proposal_feat = self.head(pooled_feat).squeeze(2).squeeze(2)
            features = F.normalize(self.feature(proposal_feat)).cpu()
            result_list = input_list.copy()
            for bi, feat in enumerate(features):
                result_list[bi]["query"]["feat"] = feat
            return result_list
        else:
            proposals, rpn_loss_cls, rpn_loss_bbox = self.rpn(base_feat, img_info, None)
            # RoI pooling based on region proposals
            pooled_feat = self.roi_pool(base_feat, proposals)

            # Extract the features of proposals
            proposal_feat = self.head(pooled_feat).squeeze(2).squeeze(2)

            scores = self.cls_score(proposal_feat)
            probs = F.softmax(scores, dim=1)
            proposal_deltas = self.bbox_pred(proposal_feat)
            features = F.normalize(self.feature(proposal_feat))

            # Unscale proposals back to raw image space
            proposals = proposals[:, 1:5] / scale
            # Unnormalize proposal deltas
            num_classes = proposal_deltas.shape[1] // 4
            stds = (
                torch.Tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
                .repeat(num_classes)
                .to(self.device)
            )
            means = (
                torch.Tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                .repeat(num_classes)
                .to(self.device)
            )
            proposal_deltas = proposal_deltas * stds + means
            # Apply proposal regression deltas
            boxes = bbox_transform_inv(proposals, proposal_deltas)
            boxes = clip_boxes(boxes, org_hws)

            # Remove those boxes with scores below the threshold
            j = 1  # Only consider foreground class
            keep = torch.nonzero(probs[:, j] > threshold)[:, 0]
            boxes = boxes[keep, j * 4 : (j + 1) * 4]
            probs = probs[keep, j]
            features = features[keep]

            # Remove redundant boxes with NMS
            detections = torch.cat((boxes, probs.unsqueeze(1)), dim=1)
            keep = nms(boxes, probs, cfg.TEST.NMS)
            detections = detections[keep]
            features = features[keep]
            return_result = {
                "pred_boxes": boxes.unsqueeze(0),
                "pred_scores": probs[..., None].unsqueeze(0),
                "reid_feats": features.unsqueeze(0),
            }
            return return_result

    def _resume_from_ckpt(self, ckpt_path, optimizer=None, lr_scheduler=None):
        if not os.path.exists(ckpt_path) or len(ckpt_path) == 0:
            logger.info(f"No checkpoint found at {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["model"], strict=False)
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(ckpt["scheduler"])
        logger.info(f"loaded checkpoint {ckpt_path}")
        logger.info(f"model was trained for {ckpt['epoch']} epochs")
