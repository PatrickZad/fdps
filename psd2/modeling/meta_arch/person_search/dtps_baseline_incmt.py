from ..build import META_ARCH_REGISTRY
from .ddetr_ps_baseline import DDETR_PS_Baseline,F,NestedTensor,_inverse_sigmoid,box_cxcywh_to_xyxy
import logging
from psd2.config import configurable
import torch
from psd2.modeling.matcher import DDetrHungarianMatcher as Matcher
from psd2.modeling.transformer import PromptDeformableTransformer
from psd2.modeling.reid_heads import build_reid_head
from ..build import META_ARCH_REGISTRY
from psd2.config import configurable
from psd2.modeling.position_encoding import PositionEmbeddingSine
from psd2.layers.set_criterion import DDetrSetCriterion as SetCriterion
import torch.nn as nn
import copy

logger = logging.getLogger("psd2")


@META_ARCH_REGISTRY.register()
class Prompt_DTPS_Baseline(DDETR_PS_Baseline):
    @configurable
    def __init__(self,*args,**kws) -> None:
        super().__init__(*args,**kws)
        hidden_dim = self.transformer.d_model
        self.prompt_query_embed = nn.ModuleList([nn.Embedding(self.cfg.DETECTOR.MODEL.INCMT.QUERY_PROMPT.NUM_PROMPT, hidden_dim * 2) for i in range(self.cfg.INCMT.TASK_ID)])
        self.det_topk=self.cfg.DETECTOR.MODEL.INCMT.OUT_TOPK
    @classmethod
    def from_config(cls, cfg):
        det_cfg = cfg.DETECTOR
        detr_cfg = det_cfg.MODEL
        trans_cfg = detr_cfg.D_TRANSFORMER
        N_steps = trans_cfg.HIDDEN_DIM // 2
        pos_enc = PositionEmbeddingSine(N_steps, normalize=True)
        transformer = PromptDeformableTransformer(
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
            task_id=cfg.INCMT.TASK_ID,
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
    def load_state_dict(self, *args, **kws):
        output = super().load_state_dict(*args, **kws)  #
        # "tail",
        for param_k in output.missing_keys:
            if "ulb_layer.queue" in param_k:
                for oim_layer in self.reid_head.loss_layers:
                    oim_layer.ulb_layer.tail = torch.tensor([0], device=self.device)
                break
        if hasattr(self.reid_head,"_param_setup"):
            self.reid_head._param_setup()
        return output

    def get_det_pred(self,img,img_whwh):
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
            query_embeds_ps = [mdl.weight for mdl in self.prompt_query_embed]
        (
            aux_info,
            hs,
            init_reference,
            inter_references,
            memory,
        ) = self.transformer(srcs, masks, pos, query_embeds,query_embeds_ps)

        outputs_classes = []
        outputs_coords = []
        outputs_rpts = []
        outputs_embs = []
        # NOTE to verify inc dets
        len_det_q=self.query_embed.weight.shape[0]
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
                    (reference[..., :2].sigmoid() * img_whwh[:, None, :2])[:,len_det_q:]
                )  # rel in aug (before padding)->abs in aug
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                outputs_rpts.append((reference.sigmoid() * img_whwh[:, None, :2])[:,len_det_q:])
            outputs_coord = tmp.sigmoid()  # B x N x 4
            outputs_coord = outputs_coord * img_whwh[:, None, :]  # ccwh_abs
            bimg, nq = outputs_coord.shape[:2]
            outputs_coord = box_cxcywh_to_xyxy(outputs_coord.flatten(0, 1)).reshape(
                bimg, nq, -1
            )
            outputs_classes.append(outputs_class[:,len_det_q:])
            outputs_coords.append(outputs_coord[:,len_det_q:])  # xyxy_abs
            outputs_embs.append(hs[lvl][:,len_det_q:])

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_rpts = torch.stack(outputs_rpts)
        outputs_embs = torch.stack(outputs_embs)
        # outputs_id_feats = F.normalize(hs, dim=-1)
        # outputs_id_feats = hs
        return (
            aux_info,
            memory,
            hs[-1][:,len_det_q:],
            (outputs_class, outputs_coord, outputs_rpts, outputs_embs),
        )

@META_ARCH_REGISTRY.register()
class Transfer_DTPS_Baseline(DDETR_PS_Baseline):
    def load_state_dict(self, *args, **kws):
        output = super().load_state_dict(*args, **kws)
        for param_k in output.missing_keys:
            if "ulb_layer.queue" in param_k:
                for oim_layer in self.reid_head.loss_layers:
                    oim_layer.ulb_layer.tail = torch.tensor([0], device=self.device)
                break
        if hasattr(self.reid_head,"_param_setup"):
            self.reid_head._param_setup()
        return output

    