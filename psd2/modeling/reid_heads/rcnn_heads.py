from mimetypes import init
from psd2.modeling.roi_heads import Res5ROIHeads
from psd2.config import configurable
from typing import Dict, List, Optional, Tuple
import torch.nn.init as init
import torch
from psd2.layers import ShapeSpec
from torch import nn
from ..poolers import ROIPooler
import inspect
import logging
from psd2.modeling.roi_heads import (
    FastRCNNOutputLayersNorm,
    FastRCNNOutputLayersNormRegOnly,
)
from psd2.layers.mem_matching_losses import build_loss_layer
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class Res5OIMHead(Res5ROIHeads):
    @configurable
    def __init__(
        self,
        *,
        oim_layer,
        in_features_reid: List[str],
        reid_emb_net,
        reid_loss_weights,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.oim_loss = oim_layer  # OIM
        self.in_features_reid = in_features_reid
        # idnet
        self.idnet = reid_emb_net
        self.in_features_reid = in_features_reid
        self.reid_loss_weights = reid_loss_weights

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super(Res5ROIHeads,cls).from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # Compatbility with old moco code. Might be useful.
        # See notes in StandardROIHeads.from_config
        if not inspect.ismethod(cls._build_res5_block):
            logger.warning(
                "The behavior of _build_res5_block may change. "
                "Please do not depend on private methods."
            )
            cls._build_res5_block = classmethod(cls._build_res5_block)

        ret["res5"], out_channels = cls._build_res5_block(cfg)
        ret["box_predictor"] = FastRCNNOutputLayersNorm(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )
        # oim layer
        ret["oim_layer"] = build_loss_layer(
            cfg.REID_HEAD.LOSS, cfg.REID_HEAD.PERSON_FEATURE.DIM
        )
        ret["in_features_reid"] = cfg.REID_HEAD.IN_FEATURES
        in_channels_reid = cfg.REID_HEAD.IN_CHANNELS
        out_dim = cfg.REID_HEAD.PERSON_FEATURE.DIM // len(cfg.REID_HEAD.IN_FEATURES)
        reid_embs = []
        for in_chnls in in_channels_reid:
            emb = nn.Sequential(nn.Linear(in_chnls, out_dim), nn.BatchNorm1d(out_dim))
            init.normal_(emb[0].weight, std=0.01)
            init.normal_(emb[1].weight, std=0.01)
            init.constant_(emb[0].bias, 0)
            init.constant_(emb[1].bias, 0)
            reid_embs.append(emb)
        ret["reid_emb_net"] = nn.Sequential(*reid_embs)
        ret["reid_loss_weights"] = {"oim": cfg.REID_HEAD.LOSS.LOSS_WEIGHTS.OIM}
        return ret

    def reid_feat(self, in_features):
        outputs = []
        for i, in_name in enumerate(self.in_features_reid):
            outputs.append(self.idnet[i](in_features[in_name]))
        return torch.cat(outputs, dim=1)

    def forward(self, features, proposals, targets=None):
        # NOTE use pid as class id in annotation for compatibility and starts with 2 for lb, 0 for ulb
        """
        See :meth:`ROIHeads.forward`.
        """

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            # 1 for background, 0 for ulb person, >1 for lb person, -1 for det ignored
            # for det loss
            reid_pids = []
            for prop in proposals:
                reid_ids = prop.gt_classes.clone()
                reid_ids[reid_ids == -1] = -2
                reid_ids[reid_ids == 1] = -2
                reid_ids[reid_ids == 0] = -1
                reid_ids[reid_ids > 1] -= 2
                reid_pids.append(reid_ids)
                prop.gt_classes[prop.gt_classes > 1] = 0  # for det classification

        proposal_boxes = [x.proposal_boxes for x in proposals]
        res4_features = self.pooler(
            [features[f] for f in self.in_features], proposal_boxes
        )
        res4_embs = F.adaptive_max_pool2d(
            res4_features, 1
        )  # res4_features.mean(dim=[2, 3])
        res4_embs = res4_embs.view(*res4_embs.shape[:-2])
        res5_features = self.res5(res4_features)
        res5_embs = F.adaptive_max_pool2d(
            res5_features, 1
        )  # res5_features.mean(dim=[2, 3])
        res5_embs = res5_embs.view(*res5_embs.shape[:-2])
        predictions = self.box_predictor(res5_embs)  # scores, proposal_deltas
        if self.training:
            losses = self.box_predictor.losses(
                predictions, proposals
            )  # "loss_cls","loss_box_reg"
            # use only foreground features for reid net
            cat_reid_ids = torch.cat(reid_pids, dim=0)
            fg_mask = cat_reid_ids > -3  # fg_mask = cat_reid_ids > -2
            fg_res4, fg_res5 = (
                res4_embs[fg_mask],
                res5_embs[fg_mask],
            )
            fg_reid_feats = self.reid_feat({"res4": fg_res4, "res5": fg_res5})
            oim_loss = self.oim_loss(fg_reid_feats, cat_reid_ids[fg_mask], None)
            losses["loss_oim"] = self.reid_loss_weights["oim"] * oim_loss["loss_oim"]
            # for vis
            with torch.no_grad():
                pred_instances, keep_inds = self.box_predictor.inference(
                    predictions, proposals
                )
                outputs = {"pred_boxes": [], "pred_scores": [], "assign_ids": []}
                for pred_img, inds_img, match_pids in zip(
                    pred_instances, keep_inds, reid_pids
                ):
                    boxes = pred_img.pred_boxes.tensor
                    scores = pred_img.scores
                    outputs["pred_boxes"].append(boxes)
                    outputs["pred_scores"].append(scores.unsqueeze(1))
                    outputs["assign_ids"].append(match_pids[inds_img])
            return outputs, losses
        else:
            reid_feats = self.reid_feat({"res4": res4_embs, "res5": res5_embs})
            props_per_img = [props.tensor.shape[0] for props in proposal_boxes]
            reid_feats_per_img = torch.split(reid_feats, props_per_img, dim=0)
            pred_instances, keep_inds = self.box_predictor.inference(
                predictions, proposals
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            outputs = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pred_img, pfeats_img, inds_img in zip(
                pred_instances, reid_feats_per_img, keep_inds
            ):
                boxes = pred_img.pred_boxes.tensor
                scores = pred_img.scores
                pfeats = pfeats_img[inds_img]
                outputs["pred_boxes"].append(boxes)
                outputs["pred_scores"].append(scores.unsqueeze(1))
                outputs["reid_feats"].append(pfeats)
            return outputs, {}


class Res5NAEHead(Res5OIMHead):
    @configurable()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_rescaler = nn.BatchNorm1d(1)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = Res5OIMHead.from_config(cfg, input_shape)
        ret["box_predictor"] = FastRCNNOutputLayersNormRegOnly(
            cfg, ShapeSpec(channels=2048, height=1, width=1)  # NOTE channel
        )
        return ret

    def reid_feat(self, in_features):
        outputs = []
        for i, in_name in enumerate(self.in_features_reid):
            outputs.append(self.idnet[i](in_features[in_name]))
        embs = torch.cat(outputs, dim=1)
        norms = embs.norm(2, 1, keepdim=True)
        embs = embs / norms.expand_as(embs).clamp(min=1e-12)
        norms = self.norm_rescaler(norms).squeeze(1)
        return embs, norms

    def forward(self, features, proposals, targets=None):
        # NOTE use pid as class id in annotation for compatibility and starts with 2 for lb, 0 for ulb
        """
        See :meth:`ROIHeads.forward`.
        """

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            # 1 for background, 0 for ulb person, >1 for lb person, -1 for det ignored
            # for det loss
            reid_pids = []
            for prop in proposals:
                reid_ids = prop.gt_classes.clone()
                reid_ids[reid_ids == -1] = -2
                reid_ids[reid_ids == 1] = -2
                reid_ids[reid_ids == 0] = -1
                reid_ids[reid_ids > 1] -= 2
                reid_pids.append(reid_ids)
                prop.gt_classes[prop.gt_classes > 1] = 0  # for det classification

        proposal_boxes = [x.proposal_boxes for x in proposals]
        res4_features = self.pooler(
            [features[f] for f in self.in_features], proposal_boxes
        )
        res4_embs = F.adaptive_max_pool2d(
            res4_features, 1
        )  # res4_features.mean(dim=[2, 3])
        res4_embs = res4_embs.view(*res4_embs.shape[:-2])
        res5_features = self.res5(res4_features)
        res5_embs = F.adaptive_max_pool2d(
            res5_features, 1
        )  # res5_features.mean(dim=[2, 3])
        res5_embs = res5_embs.view(*res5_embs.shape[:-2])
        predictions = self.box_predictor(res5_embs)  # scores, proposal_deltas
        if self.training:
            losses = self.box_predictor.losses(
                predictions, proposals
            )  # "loss_cls","loss_box_reg"
            reid_feats, norm_logits = self.reid_feat(
                {"res4": res4_embs, "res5": res5_embs}
            )
            gt_classes = [
                torch.cat([p.gt_classes for p in proposals], dim=0)
                if len(proposals)
                else torch.empty(0)
            ]
            gt_labels = torch.cat(gt_classes, dim=0)
            losses["loss_cls"] = F.binary_cross_entropy_with_logits(
                norm_logits, gt_labels.float()
            )
            # use only foreground features for reid net
            cat_reid_ids = torch.cat(reid_pids, dim=0)
            fg_mask = cat_reid_ids > -3  # NOTE fg_mask = cat_reid_ids > -2

            oim_loss = self.oim_loss(reid_feats[fg_mask], cat_reid_ids[fg_mask], None)
            losses["loss_oim"] = self.reid_loss_weights["oim"] * oim_loss["loss_oim"]
            # for vis
            with torch.no_grad():
                props_per_img = [props.tensor.shape[0] for props in proposal_boxes]
                norm_score = norm_logits.sigmoid()
                norm_scores_per_img = torch.split(norm_score, props_per_img, dim=0)
                for prop_inst, ns in zip(proposals, norm_scores_per_img):
                    prop_inst.objectness_logits = ns
                pred_instances, keep_inds = self.box_predictor.inference(
                    predictions, proposals
                )
                outputs = {"pred_boxes": [], "pred_scores": [], "assign_ids": []}
                for pred_img, inds_img, match_pids in zip(
                    pred_instances, keep_inds, reid_pids
                ):
                    boxes = pred_img.pred_boxes.tensor
                    scores = pred_img.scores
                    outputs["pred_boxes"].append(boxes)
                    outputs["pred_scores"].append(scores.unsqueeze(1))
                    outputs["assign_ids"].append(match_pids[inds_img])
            return outputs, losses
        else:
            reid_feats, norm_logits = self.reid_feat(
                {"res4": res4_embs, "res5": res5_embs}
            )
            props_per_img = [props.tensor.shape[0] for props in proposal_boxes]
            reid_feats_per_img = torch.split(reid_feats, props_per_img, dim=0)
            norm_score = norm_logits.sigmoid()
            norm_scores_per_img = torch.split(norm_score, props_per_img, dim=0)
            for prop_inst, ns in zip(proposals, norm_scores_per_img):
                    prop_inst.objectness_logits = ns
            pred_instances, keep_inds = self.box_predictor.inference(
                predictions, proposals
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            outputs = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pred_img, pfeats_img, inds_img in zip(
                pred_instances, reid_feats_per_img, keep_inds
            ):
                boxes = pred_img.pred_boxes.tensor
                scores = pred_img.scores
                pfeats = pfeats_img[inds_img]
                outputs["pred_boxes"].append(boxes)
                outputs["pred_scores"].append(scores.unsqueeze(1))
                outputs["reid_feats"].append(pfeats)
            return outputs, {}
