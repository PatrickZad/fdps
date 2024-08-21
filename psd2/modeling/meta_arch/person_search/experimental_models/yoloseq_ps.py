from ..build import META_ARCH_REGISTRY
from psd2.config import configurable
from .base import SearchBase
import torch.nn as nn
from psd2.layers.set_criterion import OIMSetCriterion as SetCriterion
from psd2.modeling.matcher import HungarianMatcher
import torch
from psd2.structures import NestedTensor
from psd2.structures.boxes import box_cxcywh_to_xyxy
from psd2.utils.events import get_event_storage
from psd2.utils.visualizer import Visualizer
import torchvision.transforms.functional as tvF
import numpy as np


@META_ARCH_REGISTRY.register()
class YOLO_SEQ_PS(SearchBase):
    @configurable
    def __init__(
        self, *, cfg, num_det_tokens, init_pe_size, mid_pe_size, use_chkp, criterion
    ) -> None:
        super().__init__(cfg)

        self.backbone.finetune_det(
            det_token_num=num_det_tokens,
            img_size=init_pe_size,
            mid_pe_size=mid_pe_size,
            use_checkpoint=use_chkp,
        )
        hidden_dim = self.backbone.output_shape()[
            self.backbone._out_features[-1]
        ].channels
        self.class_embed = MLP(hidden_dim, hidden_dim, 1, 3)  # use focal loss for cls
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.criterion = criterion

    @classmethod
    def from_config(cls, cfg):
        search_cfg = cfg.MODEL.SEARCH
        yoloseq_cfg = search_cfg.YOLOSEQ
        det_token_num = yoloseq_cfg.NUM_DET_TOKENS
        init_pe_size = yoloseq_cfg.INIT_PE_SIZE
        mid_pe_size = yoloseq_cfg.MID_PE_SIZE
        use_chkp = yoloseq_cfg.USE_CHKP
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
        weight_dict["loss_oim"] = search_cfg.PERSON_FEAT.OIM.LOSS_WEIGHT
        losses = ["labels", "boxes"]
        criterion = SetCriterion(
            1,
            matcher,
            weight_dict,
            losses,
            search_cfg.PERSON_FEAT,
            focal_alpha=loss_w_cfg.FOCAL_ALPHA,
        )
        return {
            "cfg": cfg,
            "num_det_tokens": det_token_num,
            "init_pe_size": init_pe_size,
            "mid_pe_size": mid_pe_size,
            "use_chkp": use_chkp,
            "criterion": criterion,
        }

    def get_prediction(self, img_nested_tensor):
        x = self.backbone(img_nested_tensor.tensors)
        # x = x[:, 1:,:]
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()
        out = {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
            "reid_feats": x,
        }
        return out

    def losses(self, prediction, annos):
        return self.criterion(prediction, annos)

    def run_iter(self, input_batches):
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
        prediction_dict = self.get_prediction(img_nested_tensor)
        if not self.training:
            aug_hws = prediction_dict["pred_boxes"].new_tensor(input_batches[5])
            org_hws = prediction_dict["pred_boxes"].new_tensor(input_batches[6])
            logits = prediction_dict.pop("pred_logits", None)
            prediction_dict["pred_scores"] = logits.sigmoid()
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

            return prediction_dict

        loss_dict = self.criterion(prediction_dict, annos)
        cls_error = loss_dict.pop("class_error", None)
        cardinality = loss_dict.pop("cardinality_error", None)
        if get_event_storage().iter % self.vis_period == 0:
            self.visualize_training(
                input_batches,
                prediction_dict,
                {"cls error": cls_error},
            )
        lwd = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in lwd:
                loss_dict[k] *= lwd[k]
        return loss_dict

    def forward_return_attention(self, img_nested_tensor):
        attention = self.backbone(img_nested_tensor, return_attention=True)
        return attention

    def visualize_training(self, batched_inputs, batched_dets, is_ccwh=True):
        """
        Args:
            batched_inputs:
                [imgs nested tensor, imgs paths, imgs ids, imgs bboxes, imgs person ids]
            featmap: a batched images feature map tensor
            featmap: (multi level) image feature map(s)
            batched_dets:
                {
                    "pred_logits": batched person scores B x N x 1,
                    "pred_boxes": batched cx_cy_w_h bboxes in [0,1],
                    "pred_ref_pts": batched decoder query refrence points in [0,1],
                    "reid_feat": reid feature,
                    "assign_ids": ids assigned to each reid feature
                    "aux_outputs": optional,
                    "enc_outputs": optional
                ]
            vals: scalars to be visualized
            thred: cls score threshold
        """
        from .base import COLORS, T_COLORS_BG

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
                    "boxes": bboxes,  # cx,cy,w,h abs
                    "ids": ids,
                }
            )
        bs = len(annos)
        for bi in range(bs):
            img_norm_t = samples.tensors[bi].cpu()
            img_t = trans_t2img_t(img_norm_t)
            img_rgb = img_t2rgb(img_t)
            visualize_org = Visualizer(img_rgb.copy())
            boxes = annos[bi]["boxes"].cpu()
            if is_ccwh:
                boxes = box_cxcywh_to_xyxy(boxes)
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
            img_h, img_w = img_norm_t.shape[-2:]
            boxes = batched_dets["pred_boxes"][bi].detach().cpu()
            boxes = boxes * boxes.new_tensor(((img_w, img_h) * 2,))
            if is_ccwh:
                boxes = box_cxcywh_to_xyxy(boxes)
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
                assigned_id = batched_dets["assign_ids"][bi][i].item()
                if assigned_id == -2:
                    continue
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
                    str(assigned_id),
                    boxes[i][:2],
                    horizontal_alignment="left",
                    color=t_clr,
                    bg_color=b_clr,
                )
            rgb_run_vis = [vis.get_output().get_image() for vis in visualize_runs]
            rgb_run_vis = np.concatenate(rgb_run_vis, axis=1)
            t_run_vis = tvF.to_tensor(rgb_run_vis)
            storage.put_image("img_{}/det".format(bi), t_run_vis)


import torch.nn.functional as tF


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
            x = tF.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
