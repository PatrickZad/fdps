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
from turtle import forward
import torch
import torch.nn.functional as F
from torch import nn
import math
from psd2.structures.nested_tensor import NestedTensor
from psd2.structures.nested_tensor import nested_collate_fn_idvi as nested_collate_fn
from psd2.modeling.matcher import DDetrHungarianMatcher as Matcher
from .base import SearchBase

from psd2.modeling.transformer import DeformableTransformer
import copy
from psd2.modeling.reid_heads import build_reid_head
from ..build import META_ARCH_REGISTRY
from psd2.config import configurable
from psd2.modeling.position_encoding import PositionEmbeddingSine

from psd2.utils.events import get_event_storage
from psd2.layers.set_criterion import DDetrSetCriterion as SetCriterion
from psd2.structures.boxes import box_cxcywh_to_xyxy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@META_ARCH_REGISTRY.register()
class DDETR_PS_Baseline(SearchBase):
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
        reid_head,
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
        self.reid_head = reid_head
        # self.det_topk_to_head = cfg.DETECTOR.DET_TOPK_TO_HEAD
        self.cws = cfg.CWS

    @classmethod
    def from_config(cls, cfg):
        det_cfg = cfg.DETECTOR
        detr_cfg = det_cfg.MODEL
        trans_cfg = detr_cfg.D_TRANSFORMER
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
            num_feature_levels=detr_cfg.NUM_FEAT_LEVELS,
            dec_n_points=trans_cfg.DEC_N_POINTS,
            enc_n_points=trans_cfg.ENC_N_POINTS,
            two_stage=detr_cfg.TWO_STAGE,
            two_stage_num_proposals=detr_cfg.N_QUERIES,
        )
        # Loss parameters:
        loss_cfg = det_cfg.LOSS
        loss_weights = loss_cfg.LOSS_WEIGHTS
        class_weight = loss_weights.CLS
        giou_weight = loss_weights.BOX_GIOU
        l1_weight = loss_weights.BOX_L1
        no_object_weight = loss_weights.NO_OBJECT
        deep_supervision = loss_cfg.DEEP_SUPERVISION
        use_focal = loss_cfg.FOCAL.USE_FOCAL
        focal_alpha = loss_cfg.FOCAL.ALPHA
        focal_gamma = loss_cfg.FOCAL.GAMMA
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
            for i in range(trans_cfg.DEC_DEPTH - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f"_enc": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes"]
        criterion = SetCriterion(
            num_classes=det_cfg.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            use_focal=use_focal,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        # postprocessors = {"bbox": PostProcess()}
        reid_head = build_reid_head(cfg.REID_HEAD)
        return {
            "cfg": cfg,
            "pos_encoding": pos_enc,
            "transformer": transformer,
            "in_feats": detr_cfg.IN_FEATS,
            "num_feat_levels": detr_cfg.NUM_FEAT_LEVELS,
            "n_queries": detr_cfg.N_QUERIES,
            "use_aux_loss": deep_supervision,
            "with_box_refine": detr_cfg.BOX_REFINE,
            "two_stage": detr_cfg.TWO_STAGE,
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

    def get_det_pred(self, img: NestedTensor, img_whwh):
        # nestedtensor to resnet backbone and position encodings
        xs = self.backbone(img.tensors)
        features = []
        pos = []
        for name, x in xs.items():
            m = img.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            nested_feat = NestedTensor(x, mask)
            features.append(nested_feat)
            pos.append(self.pos_enc(nested_feat))
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
                m = img.mask
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
            aux_info,
            hs,
            init_reference,
            inter_references,
            memory,
        ) = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        outputs_rpts = []
        outputs_embs = []
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
                outputs_rpts.append(
                    reference[..., :2].sigmoid() * img_whwh[:, None, :2]
                )  # rel in aug (before padding)->abs in aug
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                outputs_rpts.append(reference.sigmoid() * img_whwh[:, None, :2])
            outputs_coord = tmp.sigmoid()  # B x N x 4
            outputs_coord = outputs_coord * img_whwh[:, None, :]  # ccwh_abs
            bimg, nq = outputs_coord.shape[:2]
            outputs_coord = box_cxcywh_to_xyxy(outputs_coord.flatten(0, 1)).reshape(
                bimg, nq, -1
            )
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)  # xyxy_abs
            outputs_embs.append(hs[lvl])

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_rpts = torch.stack(outputs_rpts)
        outputs_embs = torch.stack(outputs_embs)
        # outputs_id_feats = F.normalize(hs, dim=-1)
        # outputs_id_feats = hs
        return (
            aux_info,
            memory,
            hs[-1],
            (outputs_class, outputs_coord, outputs_rpts, outputs_embs),
        )

    def forward(self, input_list):
        if "query" in input_list[0].keys():
            return input_list
        input_batches = self.preprocess_input(input_list)
        img_nested_tensor: NestedTensor = input_batches[0]
        images_whwh = img_nested_tensor.tensors.new_tensor(
            [hw[::-1] * 2 for hw in img_nested_tensor.image_sizes]
        )
        targets = self.prepare_targets(input_batches)
        (
            aux_info,
            enc_memory,
            dec_hs,
            (outputs_class, outputs_coord, outputs_rpts, outputs_embs),
        ) = self.get_det_pred(img_nested_tensor, images_whwh)
        all_output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_embs": outputs_embs[-1],
            "pred_ref_pts": outputs_rpts[-1],  # for vis only
        }  # xyxy boxes in aug
        all_output["aux_outputs"] = [
            {"pred_logits": a, "pred_boxes": b, "pred_embs": c, "pred_ref_pts": d}
            for a, b, c, d in zip(
                outputs_class[:-1],
                outputs_coord[:-1],
                outputs_embs[:-1],
                outputs_rpts[:-1],
            )
        ]
        if self.reid_head.__class__.__name__.find("X") != -1:
            num_reid_layers = len(self.reid_head.loss_layers)
            num_det_layers = outputs_class.shape[0] - num_reid_layers
            det_output = {}
            det_output = {
                k: all_output["aux_outputs"][-1][k]
                for k in all_output["aux_outputs"][-1]
            }
            det_output["aux_outputs"] = []
            # before X
            for xitem in all_output["aux_outputs"][: num_det_layers - num_reid_layers]:
                det_output["aux_outputs"].append({k: xitem[k] for k in xitem})
            # after X
            for xitem in all_output["aux_outputs"][
                num_det_layers - num_reid_layers : -1 : 2
            ]:
                det_output["aux_outputs"].append({k: xitem[k] for k in xitem})
        else:
            det_output = all_output
        (
            spatial_shapes,
            level_start_index,
            valid_ratios,
            mask_flatten,
        ) = aux_info  # src is from conv
        if self.training:
            # det losses
            match_ids, loss_dict = self.criterion(det_output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            # reid losses
            bs, _, c = all_output["pred_embs"].shape
            query_embed, _ = torch.split(self.query_embed.weight, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            if hasattr(self, "prompt_query_embed"):
                q_embs = [query_embed]
                for q_emb_ps in self.prompt_query_embed:
                    query_embed_p, _ = torch.split(q_emb_ps.weight, c, dim=1)
                    query_embed_p = query_embed_p.unsqueeze(0).expand(bs, -1, -1)
                    q_embs.append(query_embed_p)
                query_embed = torch.cat(q_embs, dim=1)
            memory_src = enc_memory
            reid_head_outputs = self.reid_head(
                enc_memory=memory_src,
                det_outputs=all_output,
                targets=targets,
                det_match_indices=match_ids,
                queries=dec_hs,
                query_pos=query_embed,
                pts_valid_ratios=valid_ratios,
                memory_spatial_shapes=spatial_shapes,
                memory_level_start_index=level_start_index,
                memory_mask_flatten=mask_flatten,
            )
            reid_losses = reid_head_outputs["losses"]
            loss_dict.update(reid_losses)
            # for visualization
            # output["reid_feats"] = reid_head_outputs["reid_feats"]
            det_output["assign_ids"] = reid_head_outputs["assign_ids"]

            """if "aux_outputs" in reid_head_outputs:
                for idx, item in enumerate(reid_head_outputs["aux_outputs"]):
                    output["aux_outputs"][idx].update(item)"""

            if get_event_storage().iter % self.vis_period == 0:
                # reshape memory to restore image features for visualization
                img_trans_feats = []
                if isinstance(enc_memory, list):
                    h, w = spatial_shapes[0]
                    for mem in enc_memory:
                        img_trans_feats.append(
                            mem.transpose(1, 2).reshape(bs, -1, h, w)
                        )
                else:
                    if level_start_index.shape[0] > 1:
                        feat_flatten_lens = (
                            torch.cat(
                                (
                                    level_start_index[1:],
                                    level_start_index.new_tensor([enc_memory.shape[1]]),
                                )
                            )
                            - level_start_index
                        )
                        level_feat_flattens = torch.split(
                            enc_memory, feat_flatten_lens.tolist(), dim=1
                        )
                        for lvl, flatten_feat in enumerate(level_feat_flattens):
                            h, w = spatial_shapes[lvl]
                            img_trans_feats.append(
                                flatten_feat.transpose(1, 2).reshape(bs, -1, h, w)
                            )
                    else:
                        h, w = spatial_shapes[0]
                        img_trans_feats.append(
                            enc_memory.transpose(1, 2).reshape(bs, -1, h, w)
                        )
                self.visualize_training(
                    input_batches, img_trans_feats, det_output, is_ccwh=False
                )
            return loss_dict
        else:
            # remove unused
            aug_hws = det_output["pred_boxes"].new_tensor(input_batches[5])
            org_hws = det_output["pred_boxes"].new_tensor(input_batches[6])  # B x 2
            org_whwh = torch.stack([org_hws[:, 1], org_hws[:, 0]] * 2, dim=1)  # B x 4
            aug_whwh = torch.stack([aug_hws[:, 1], aug_hws[:, 0]] * 2, dim=1)  # B x 4
            bs, _, c = all_output["pred_embs"].shape
            query_embed, _ = torch.split(self.query_embed.weight, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            if hasattr(self, "prompt_query_embed"):
                q_embs = []  # [query_embed]
                for q_emb_ps in self.prompt_query_embed:
                    query_embed_p, _ = torch.split(q_emb_ps.weight, c, dim=1)
                    query_embed_p = query_embed_p.unsqueeze(0).expand(bs, -1, -1)
                    q_embs.append(query_embed_p)
                query_embed = torch.cat(q_embs, dim=1)
            reid_feats = self.reid_head(
                enc_memory=enc_memory,
                det_outputs=all_output,
                targets=None,
                det_match_indices=None,
                queries=dec_hs,
                query_pos=query_embed,
                pts_valid_ratios=valid_ratios,
                memory_spatial_shapes=spatial_shapes,
                memory_level_start_index=level_start_index,
                memory_mask_flatten=mask_flatten,
                img_aug_whwh=aug_whwh,
            )["reid_feats"]
            logits = det_output.pop("pred_logits", None)
            det_output["pred_scores"] = logits.sigmoid()
            sf = org_whwh / aug_whwh
            # back to org abs
            det_output["pred_boxes"] *= sf.unsqueeze(1)  # B x 1 x 4
            # NOTE feat norm performed in reid head
            if self.cws:
                det_output["reid_feats"] = reid_feats * det_output["pred_scores"]
            else:
                det_output["reid_feats"] = reid_feats
            det_output.pop("aux_outputs", None)
            det_output.pop("pred_ref_pts", None)
            det_output.pop("pred_embs", None)
            return det_output


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


import psd2.layers as layers


@META_ARCH_REGISTRY.register()
class PSTR_Baseline(DDETR_PS_Baseline):
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
        reid_head,
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
        # 3 levels for reid, the mid shared with det, all resize to 1/16
        num_backbone_outs = len(in_feats)
        assert num_backbone_outs == 3
        input_proj_list = []
        out_proj_list = []
        for i, _ in enumerate(in_feats[::-1]):
            in_channels = output_shape[_].channels
            input_proj_list.append(
                layers.Conv2d(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    padding=0,
                    norm=None,
                    activation=None,
                )
            )
            out_proj_list.append(
                layers.DeformConvPack(
                    hidden_dim, hidden_dim, 3, padding=1, stride=2 if i == 2 else 1
                )
            )
        self.input_proj = nn.ModuleList(input_proj_list)
        self.out_proj = nn.ModuleList(out_proj_list)
        self.aux_loss = use_aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(1) * bias_value  # only 1 category
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

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
        self.reid_head = reid_head
        # self.det_topk_to_head = cfg.DETECTOR.DET_TOPK_TO_HEAD
        self.cws = cfg.CWS
        # det feat level
        self.det_feat_name = cfg.DETECTOR.MODEL.DET_FEAT

    def get_det_pred(self, img: NestedTensor, img_whwh):
        # nestedtensor to resnet backbone and position encodings
        xs = self.backbone(img.tensors)
        neck_feats = {}
        feat_names = list(xs.keys())
        prev_shape = xs[feat_names[-2]].shape[2:]
        for ni, name in enumerate(feat_names[::-1]):
            lateral_feat = self.input_proj[ni](xs[name])
            if ni == 0:
                out = F.interpolate(lateral_feat, size=prev_shape, mode="nearest")
            else:
                out = lateral_feat
            out_feat = self.out_proj[ni](out)
            neck_feats[name] = out_feat
        det_features = []
        det_pos = []
        det_f = neck_feats[self.det_feat_name]
        m = img.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=det_f.shape[-2:]).to(torch.bool)[0]
        nested_feat = NestedTensor(det_f, mask)
        det_features.append(nested_feat)
        det_pos.append(self.pos_enc(nested_feat))
        srcs = []
        masks = []
        for l, feat in enumerate(det_features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        (
            aux_info,
            hs,
            init_reference,
            inter_references,
            memory,
        ) = self.transformer(srcs, masks, det_pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        outputs_rpts = []
        outputs_embs = []
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
                outputs_rpts.append(
                    reference[..., :2].sigmoid() * img_whwh[:, None, :2]
                )  # rel in aug (before padding)->abs in aug
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                outputs_rpts.append(reference.sigmoid() * img_whwh[:, None, :2])
            outputs_coord = tmp.sigmoid()  # B x N x 4
            outputs_coord = outputs_coord * img_whwh[:, None, :]  # ccwh_abs
            bimg, nq = outputs_coord.shape[:2]
            outputs_coord = box_cxcywh_to_xyxy(outputs_coord.flatten(0, 1)).reshape(
                bimg, nq, -1
            )
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)  # xyxy_abs
            outputs_embs.append(hs[lvl])

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_rpts = torch.stack(outputs_rpts)
        outputs_embs = torch.stack(outputs_embs)
        # outputs_id_feats = F.normalize(hs, dim=-1)
        # outputs_id_feats = hs
        memory = [neck_feats[fn].flatten(2).transpose(1, 2) for fn in feat_names]
        return (
            aux_info,
            memory,
            hs[-1],
            (outputs_class, outputs_coord, outputs_rpts, outputs_embs),
        )


@META_ARCH_REGISTRY.register()
class DDETR2s_PS_Baseline(DDETR_PS_Baseline):
    @configurable
    def __init__(self, *args, **kws) -> None:
        super().__init__(*args, **kws)
        assert self.two_stage

    def get_det_pred(self, img: NestedTensor, img_whwh):
        # nestedtensor to resnet backbone and position encodings
        xs = self.backbone(img.tensors)
        features = []
        pos = []
        for name, x in xs.items():
            m = img.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            nested_feat = NestedTensor(x, mask)
            features.append(nested_feat)
            pos.append(self.pos_enc(nested_feat))
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
                m = img.mask
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
            aux_info,
            hs,
            init_reference,
            inter_references,
            memory,
            query_pos,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        outputs_rpts = []
        outputs_embs = []
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
                outputs_rpts.append(
                    reference[..., :2].sigmoid() * img_whwh[:, None, :2]
                )  # rel in aug (before padding)->abs in aug
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                outputs_rpts.append(reference.sigmoid() * img_whwh[:, None, :2])
            outputs_coord = tmp.sigmoid()  # B x N x 4
            outputs_coord = outputs_coord * img_whwh[:, None, :]  # ccwh_abs
            bimg, nq = outputs_coord.shape[:2]
            outputs_coord = box_cxcywh_to_xyxy(outputs_coord.flatten(0, 1)).reshape(
                bimg, nq, -1
            )
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)  # xyxy_abs
            outputs_embs.append(hs[lvl])

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_rpts = torch.stack(outputs_rpts)
        outputs_embs = torch.stack(outputs_embs)
        # outputs_id_feats = F.normalize(hs, dim=-1)
        # outputs_id_feats = hs
        # two stage
        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        enc_outputs_coord = enc_outputs_coord * img_whwh[:, None, :]  # ccwh_abs
        bimg, nq = enc_outputs_coord.shape[:2]
        enc_outputs_coord = box_cxcywh_to_xyxy(enc_outputs_coord.flatten(0, 1)).reshape(
            bimg, nq, -1
        )
        return (
            aux_info,
            memory,
            hs[-1],
            (
                outputs_class,
                outputs_coord,
                outputs_rpts,
                outputs_embs,
                query_pos,
                enc_outputs_class,
                enc_outputs_coord,
            ),
        )

    def forward(self, input_list):
        if "query" in input_list[0].keys():
            return input_list
        input_batches = self.preprocess_input(input_list)
        img_nested_tensor: NestedTensor = input_batches[0]
        images_whwh = img_nested_tensor.tensors.new_tensor(
            [hw[::-1] * 2 for hw in img_nested_tensor.image_sizes]
        )
        targets = self.prepare_targets(input_batches)
        (
            aux_info,
            enc_memory,
            dec_hs,
            (
                outputs_class,
                outputs_coord,
                outputs_rpts,
                outputs_embs,
                query_pos,
                enc_outputs_class,
                enc_outputs_coord,
            ),
        ) = self.get_det_pred(img_nested_tensor, images_whwh)
        all_output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_embs": outputs_embs[-1],
            "pred_ref_pts": outputs_rpts[-1],  # for vis only
        }  # xyxy boxes in aug
        all_output["aux_outputs"] = [
            {"pred_logits": a, "pred_boxes": b, "pred_embs": c, "pred_ref_pts": d}
            for a, b, c, d in zip(
                outputs_class[:-1],
                outputs_coord[:-1],
                outputs_embs[:-1],
                outputs_rpts[:-1],
            )
        ]
        if self.reid_head.__class__.__name__.find("X") != -1:
            num_reid_layers = len(self.reid_head.loss_layers)
            num_det_layers = outputs_class.shape[0] - num_reid_layers
            det_output = {}
            det_output = {
                k: all_output["aux_outputs"][-1][k]
                for k in all_output["aux_outputs"][-1]
            }
            det_output["aux_outputs"] = []
            # before X
            for xitem in all_output["aux_outputs"][: num_det_layers - num_reid_layers]:
                det_output["aux_outputs"].append({k: xitem[k] for k in xitem})
            # after X
            for xitem in all_output["aux_outputs"][
                num_det_layers - num_reid_layers : -1 : 2
            ]:
                det_output["aux_outputs"].append({k: xitem[k] for k in xitem})
        else:
            det_output = all_output
        det_output["enc_outputs"] = {
            "pred_logits": enc_outputs_class,
            "pred_boxes": enc_outputs_coord,
        }
        (
            spatial_shapes,
            level_start_index,
            valid_ratios,
            mask_flatten,
        ) = aux_info
        if self.training:
            # det losses
            match_ids, loss_dict = self.criterion(det_output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            # reid losses
            bs, _, c = all_output["pred_embs"].shape
            query_embed = query_pos
            if hasattr(self, "prompt_query_embed"):
                q_embs = [query_embed]
                for q_emb_ps in self.prompt_query_embed:
                    query_embed_p, _ = torch.split(q_emb_ps.weight, c, dim=1)
                    query_embed_p = query_embed_p.unsqueeze(0).expand(bs, -1, -1)
                    q_embs.append(query_embed_p)
                query_embed = torch.cat(q_embs, dim=1)
            memory_src = enc_memory
            to_head_out = {k: v for k, v in all_output.items() if k != "enc_outputs"}
            reid_head_outputs = self.reid_head(
                enc_memory=memory_src,
                det_outputs=to_head_out,
                targets=targets,
                det_match_indices=match_ids,
                queries=dec_hs,
                query_pos=query_embed,
                pts_valid_ratios=valid_ratios,
                memory_spatial_shapes=spatial_shapes,
                memory_level_start_index=level_start_index,
                memory_mask_flatten=mask_flatten,
            )
            reid_losses = reid_head_outputs["losses"]
            loss_dict.update(reid_losses)
            # for visualization
            # output["reid_feats"] = reid_head_outputs["reid_feats"]
            det_output["assign_ids"] = reid_head_outputs["assign_ids"]

            """if "aux_outputs" in reid_head_outputs:
                for idx, item in enumerate(reid_head_outputs["aux_outputs"]):
                    output["aux_outputs"][idx].update(item)"""

            if get_event_storage().iter % self.vis_period == 0:
                # reshape memory to restore image features for visualization
                img_trans_feats = []
                if isinstance(enc_memory, list):
                    h, w = spatial_shapes[0]
                    for mem in enc_memory:
                        img_trans_feats.append(
                            mem.transpose(1, 2).reshape(bs, -1, h, w)
                        )
                else:
                    if level_start_index.shape[0] > 1:
                        feat_flatten_lens = (
                            torch.cat(
                                (
                                    level_start_index[1:],
                                    level_start_index.new_tensor([enc_memory.shape[1]]),
                                )
                            )
                            - level_start_index
                        )
                        level_feat_flattens = torch.split(
                            enc_memory, feat_flatten_lens.tolist(), dim=1
                        )
                        for lvl, flatten_feat in enumerate(level_feat_flattens):
                            h, w = spatial_shapes[lvl]
                            img_trans_feats.append(
                                flatten_feat.transpose(1, 2).reshape(bs, -1, h, w)
                            )
                    else:
                        h, w = spatial_shapes[0]
                        img_trans_feats.append(
                            enc_memory.transpose(1, 2).reshape(bs, -1, h, w)
                        )
                self.visualize_training(
                    input_batches, img_trans_feats, det_output, is_ccwh=False
                )
            return loss_dict
        else:
            # remove unused
            aug_hws = det_output["pred_boxes"].new_tensor(input_batches[5])
            org_hws = det_output["pred_boxes"].new_tensor(input_batches[6])  # B x 2
            org_whwh = torch.stack([org_hws[:, 1], org_hws[:, 0]] * 2, dim=1)  # B x 4
            aug_whwh = torch.stack([aug_hws[:, 1], aug_hws[:, 0]] * 2, dim=1)  # B x 4
            bs, _, c = all_output["pred_embs"].shape
            query_embed = query_pos
            if hasattr(self, "prompt_query_embed"):
                q_embs = []  # [query_embed]
                for q_emb_ps in self.prompt_query_embed:
                    query_embed_p, _ = torch.split(q_emb_ps.weight, c, dim=1)
                    query_embed_p = query_embed_p.unsqueeze(0).expand(bs, -1, -1)
                    q_embs.append(query_embed_p)
                query_embed = torch.cat(q_embs, dim=1)
            to_head_out = {k: v for k, v in all_output.items() if k != "enc_outputs"}
            reid_feats = self.reid_head(
                enc_memory=enc_memory,
                det_outputs=to_head_out,
                targets=None,
                det_match_indices=None,
                queries=dec_hs,
                query_pos=query_embed,
                pts_valid_ratios=valid_ratios,
                memory_spatial_shapes=spatial_shapes,
                memory_level_start_index=level_start_index,
                memory_mask_flatten=mask_flatten,
                img_aug_whwh=aug_whwh,
            )["reid_feats"]
            logits = det_output.pop("pred_logits", None)
            det_output["pred_scores"] = logits.sigmoid()
            sf = org_whwh / aug_whwh
            # back to org abs
            det_output["pred_boxes"] *= sf.unsqueeze(1)  # B x 1 x 4
            # NOTE feat norm performed in reid head
            if self.cws:
                det_output["reid_feats"] = reid_feats * det_output["pred_scores"]
            else:
                det_output["reid_feats"] = reid_feats
            det_output.pop("aux_outputs", None)
            det_output.pop("pred_ref_pts", None)
            det_output.pop("pred_embs", None)
            det_output.pop("enc_outputs", None)
            return det_output


@META_ARCH_REGISTRY.register()
class IF_DTPS(DDETR_PS_Baseline):
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
            if "img2_input" in input_dict:
                img2_dict = input_dict
                img_paths.append(img2_dict["file_name"])
                img_names.append(img2_dict["image_id"])
                img_ts.append(img2_dict["image"].to(self.device))
                xyxy_box_t = torch.tensor(
                    img2_dict["boxes"],
                    dtype=torch.float32,
                    device=self.device,
                )  # xyxy
                img_boxes.append(xyxy_box_t)
                img_pids.append(torch.tensor(img2_dict["ids"], device=self.device))
                img_aug_hws.append((img2_dict["height"], img2_dict["width"]))
                img_org_hws.append((img2_dict["org_height"], img2_dict["org_width"]))
                img_org_boxes.append(torch.tensor(img2_dict["org_boxes"]))

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
