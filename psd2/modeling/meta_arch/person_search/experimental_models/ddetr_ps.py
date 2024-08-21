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
import torch.nn.functional as F
from torch import nn
import math


from psd2.structures import NestedTensor


from psd2.modeling.matcher import HungarianMatcher
from .base import SearchBase

from psd2.modeling.transformer import DeformableTransformer
import copy

from ..build import META_ARCH_REGISTRY
from psd2.config import configurable
from psd2.modeling.position_encoding import PositionEmbeddingSine

from psd2.utils.events import get_event_storage
from psd2.layers.set_criterion import OIMSetCriterion as SetCriterion
from psd2.structures.boxes import box_cxcywh_to_xyxy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@META_ARCH_REGISTRY.register()
class DDETR_PS(SearchBase):
    """This is the Deformable DETR module that performs person search"""

    @configurable
    def __init__(
        self,
        *,
        cfg,
        pos_encoding,
        transformer,
        in_feats,
        num_feat_levels,
        n_queries,
        use_aux_loss,
        with_box_refine,
        two_stage,
        criterion,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__(cfg)

        self.num_queries = n_queries
        self.transformer = transformer
        self.pos_enc = pos_encoding
        hidden_dim = self.transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, 1)  # only 1 category
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feat_levels
        if not two_stage:
            self.query_embed = nn.Embedding(n_queries, hidden_dim * 2)
        output_shape = self.backbone.output_shape()
        if self.num_feature_levels > 1:
            num_backbone_outs = len(in_feats)
            input_proj_list = []
            for _ in in_feats:
                in_channels = output_shape[_].channels
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feat_levels - num_backbone_outs):
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
            in_channels = output_shape[in_feats[-1]].channels
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
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(1) * bias_value  # only 1 category
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if two_stage
            else self.transformer.decoder.num_layers
        )
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
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        # for training
        # self.matcher = matcher
        # self.loss_weights = weight_dict
        self.criterion = criterion

    @classmethod
    def from_config(cls, cfg):
        search_cfg = cfg.MODEL.SEARCH
        trans_cfg = search_cfg.D_TRANSFORMER
        ddetr_cfg = search_cfg.DDETR
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
            num_feature_levels=ddetr_cfg.NUM_FEAT_LEVELS,
            dec_n_points=trans_cfg.DEC_N_POINTS,
            enc_n_points=trans_cfg.ENC_N_POINTS,
            two_stage=ddetr_cfg.TWO_STAGE,
            two_stage_num_proposals=ddetr_cfg.N_QUERIES,
        )
        mt_cfg = search_cfg.MATCHER
        matcher = HungarianMatcher(
            cost_class=mt_cfg.SET_COST_CLASS,
            cost_bbox=mt_cfg.SET_COST_BBOX,
            cost_giou=mt_cfg.SET_COST_GIOU,
        )
        loss_w_cfg = search_cfg.LOSS_WEIGHTS
        weight_dict = {
            "loss_ce": loss_w_cfg.CLS_LOSS,
            "loss_bbox": loss_w_cfg.BBOX_LOSS,
        }
        weight_dict["loss_giou"] = loss_w_cfg.GIOU_LOSS
        if ddetr_cfg.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(trans_cfg.DEC_DEPTH - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f"_enc": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        weight_dict["loss_oim"] = search_cfg.PERSON_FEAT.OIM.LOSS_WEIGHT
        if search_cfg.PERSON_FEAT.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(trans_cfg.DEC_DEPTH - 1):
                aux_weight_dict.update({"loss_oim" + f"_{i}": weight_dict["loss_oim"]})
            aux_weight_dict.update({"loss_oim" + f"_enc": weight_dict["loss_oim"]})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes"]  # , "cardinality"]   loss name for detection only
        # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
        criterion = SetCriterion(
            1,
            matcher,
            weight_dict,
            losses,
            search_cfg.PERSON_FEAT,
            focal_alpha=loss_w_cfg.FOCAL_ALPHA,
        )
        # postprocessors = {"bbox": PostProcess()}
        return {
            "cfg": cfg,
            "pos_encoding": pos_enc,
            "transformer": transformer,
            "in_feats": ddetr_cfg.IN_FEATS,
            "num_feat_levels": ddetr_cfg.NUM_FEAT_LEVELS,
            "n_queries": ddetr_cfg.N_QUERIES,
            "use_aux_loss": ddetr_cfg.AUX_LOSS,
            "with_box_refine": ddetr_cfg.BOX_REFINE,
            "two_stage": ddetr_cfg.TWO_STAGE,
            "criterion": criterion,
        }

    def conv_and_pos(self, inputs: NestedTensor):
        # nestedtensor to resnet backbone and position encodings
        xs = self.backbone(inputs.tensors)
        conv_outs = []
        pos_encs = []
        for name, x in xs.items():
            m = inputs.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]

            nested_feat = NestedTensor(x, mask)
            conv_outs.append(nested_feat)
            pos_encs.append(self.pos_enc(nested_feat))
        return conv_outs, pos_encs

    def get_prediction(self, img_nested_tensor):
        features, pos = self.conv_and_pos(img_nested_tensor)
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
                m = img_nested_tensor.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.pos_enc(NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            trans_feats,
        ) = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        outputs_rpts = []
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
                outputs_rpts.append(reference[..., :2].sigmoid())
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                outputs_rpts.append(reference.sigmoid())
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)  # cx,cy,w,h

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_rpts = torch.stack(outputs_rpts)
        # outputs_id_feats = F.normalize(hs, dim=-1)
        outputs_id_feats = hs
        out = {
            "pred_logits": outputs_class[-1],
            "pred_ref_pts": outputs_rpts[-1],
            "pred_boxes": outputs_coord[-1],
            "reid_feats": outputs_id_feats[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_rpts, outputs_id_feats
            )

        if self.two_stage:
            # TODO return reid features, ref points
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out, trans_feats

    def losses(self, prediction, annos):
        return self.criterion(prediction, annos)

    def run_iter(self, input_batches):
        """
        args:
            input_batches: [img_ts, img_paths, img_names, img_boxes, img_pids]
        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x 2]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        img_nested_tensor: NestedTensor = input_batches[0]
        annos = []
        padd_h, padd_w = img_nested_tensor.tensors.shape[-2:]
        for fname, imgid, bboxes, ids in zip(*input_batches[1:5]):
            annos.append(
                {
                    "file_name": fname,
                    "image_id": imgid,
                    "boxes": bboxes
                    / bboxes.new_tensor(((padd_w, padd_h) * 2,)),  # cx,cy,w,h in [0,1]
                    "ids": ids,
                    "labels": torch.zeros(bboxes.shape[0], dtype=torch.long).to(
                        self.device
                    ),
                }
            )
        prediction_dict, trans_feats = self.get_prediction(img_nested_tensor)
        if not self.training:
            aug_hws = prediction_dict["pred_boxes"].new_tensor(input_batches[5])
            org_hws = prediction_dict["pred_boxes"].new_tensor(input_batches[6])
            logits = prediction_dict.pop("pred_logits", None)
            prediction_dict["pred_scores"] = logits.sigmoid()
            prediction_dict.pop("pred_ref_pts")
            # B x N x 4 cx,cy,w,h in [0,1] -> abs in padded
            boxes_aug_abs = prediction_dict["pred_boxes"] * prediction_dict[
                "pred_boxes"
            ].new_tensor((((padd_w, padd_h) * 2,),))
            scale_factors = org_hws / aug_hws  # B x 2
            scale_factors = scale_factors.unsqueeze(1)  # B x 1 x 2
            scale_factors_w = scale_factors[:, :, 1:2]
            scale_factors_h = scale_factors[:, :, :1]
            # back to org abs
            factors = torch.cat([scale_factors_w, scale_factors_h] * 2, dim=-1)
            prediction_dict["pred_boxes"] = box_cxcywh_to_xyxy(boxes_aug_abs * factors)

            inter_outs = prediction_dict.pop("aux_outputs")
            feat_lvl = self.cfg.MODEL.SEARCH.PERSON_FEAT.FEAT_BASE_LVL_IDX
            if feat_lvl < len(inter_outs):
                prediction_dict["reid_feats"] = inter_outs[feat_lvl]["reid_feats"]
            return prediction_dict

        loss_dict = self.criterion(prediction_dict, annos)
        cls_error = loss_dict.pop("class_error", None)
        cardinality = loss_dict.pop("cardinality_error", None)
        if get_event_storage().iter % self.vis_period == 0:
            self.visualize_training(
                input_batches,
                trans_feats,
                prediction_dict,
                {"cls error": cls_error},
            )
        lwd = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in lwd:
                loss_dict[k] *= lwd[k]
        return loss_dict

    @torch.jit.unused
    def _set_aux_loss(
        self, outputs_class, outputs_coord, outputs_rpts, outputs_id_feats
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_ref_pts": c, "reid_feats": d}
            for a, b, c, d in zip(
                outputs_class[:-1],
                outputs_coord[:-1],
                outputs_rpts[:-1],
                outputs_id_feats[:-1],
            )
        ]


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
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
