from matplotlib.pyplot import axis
from psd2.modeling.transformer.deformable import (
    DeformableTransformerDecoderLayer,
    _get_clones,
    _get_activation_fn,
)
from .base_reid_head import ReidHeadBase
import torch
import torch.nn.functional as tF
import random
from torch import nn


sa_modes = ["original", "res_linear", "linear", "res_dynamic", "dynamic"]


class CompBN1D(nn.BatchNorm1d):
    def forward(self, x):
        """
        x: batch x seq x channel
        """
        if x.dim() == 3:
            x_t = x.transpose(1, 2)
            x_n = super().forward(x_t)
            return x_n.transpose(1, 2)
        else:
            return super().forward(x)


def get_bt_neck(type_name, in_dim):
    if type_name == "none":
        return nn.Identity()
    else:
        bt_neck = {"bn": CompBN1D, "ln": nn.LayerNorm}[type_name](in_dim)
        bt_neck.bias.requires_grad_(False)
        nn.init.constant_(bt_neck.weight, 1.0)
        nn.init.constant_(bt_neck.bias, 0.0)
        return bt_neck


class DTransDecoderXHead(ReidHeadBase):
    def __init__(self, head_cfg):
        super().__init__(head_cfg)
        self.query_gradient = head_cfg.QUERY_BP
        self.bottelnecks = nn.ModuleList()
        bt_neck_type = head_cfg.BOTTELNECK
        for li in range(len(self.loss_layers)):
            self.bottelnecks.append(get_bt_neck(bt_neck_type, self.pfeat_dim))

    def get_pfeats(
        self,
        det_outputs,
    ):
        # split det base and reid feats
        all_outputs = det_outputs["aux_outputs"]
        all_outputs.append({k: det_outputs[k] for k in det_outputs if "aux" not in k})
        num_reid_layers = len(self.loss_layers)
        num_det_layers = len(all_outputs) - num_reid_layers
        num_before_reid = num_det_layers - num_reid_layers
        det_priors = all_outputs[:num_before_reid] + all_outputs[num_before_reid::2]
        if self.query_gradient:
            reid_pfeats = [
                out["pred_embs"] for out in all_outputs[num_before_reid + 1 :: 2]
            ]
        else:
            reid_pfeats = [
                out["pred_embs"].detach()
                for out in all_outputs[num_before_reid + 1 :: 2]
            ]
        return det_priors, reid_pfeats

    def forward(self, det_outputs, targets, det_match_indices, *args, **kwargs):
        # TODO person for oim only
        # det_outputs, targets, xyxy_abs boxes
        det_priors, pfeats_out = self.get_pfeats(det_outputs)
        len_pfeats = len(pfeats_out)
        head_outputs = {}
        if self.training:
            # NOTE update to only involve boxes with valid ids
            head_outputs["losses"] = {}
            if len(self.loss_layers) > 1:  # supervise multi layers in this head
                # TODO compute only with ids
                head_aux_outputs = []
                lvl_assign_ids, lvl_as_s = [], []
                for i in range(len_pfeats):
                    assign_ids, as_s = self._id_assign_method(
                        det_priors[-(len_pfeats - i)],
                        targets,
                        det_match_indices[-(len_pfeats - i)],
                    )
                    lvl_assign_ids.append(assign_ids)
                    lvl_as_s.append(as_s)
                    if i < len_pfeats - 1:
                        head_aux_outputs.append({"assign_ids": assign_ids})
                    else:
                        head_outputs["assign_ids"] = assign_ids
                aux_losses = {}
                for i in range(len_pfeats):
                    pfeats_cls = self.bottelnecks[i](pfeats_out[i])
                    losses = self.compute_losses_lvl(
                        pfeats_cls,
                        lvl_assign_ids[i],
                        lvl_as_s[i],
                        det_priors[-(len_pfeats - i)]["pred_logits"],
                        loss_layer_lvl=i,
                    )
                    if self.metric_loss is not None:
                        # TODO choose neck feat
                        metric_losses = self.compute_metric_losses(
                            pfeats_out[i],
                            lvl_assign_ids[i],
                            lvl_as_s[i],
                            det_priors[-(len_pfeats - i)]["pred_logits"],
                        )
                        losses.update(metric_losses)
                    if i < len_pfeats - 1:
                        for k, v in losses.items():
                            aux_losses[k + "_{}".format(i)] = v
                    else:
                        head_outputs["losses"].update(losses)
                head_outputs["losses"].update(aux_losses)
                head_outputs["aux_outputs"] = head_aux_outputs

            elif self.shared_aux:
                # TODO compute only with ids
                head_aux_outputs = []
                lvl_assign_ids, lvl_as_s = [], []
                for i in range(len_pfeats):
                    assign_ids, as_s = self._id_assign_method(
                        det_priors[-(len_pfeats - i)],
                        targets,
                        det_match_indices[-(len_pfeats - i)],
                    )
                    lvl_assign_ids.append(assign_ids)
                    lvl_as_s.append(as_s)
                    head_aux_outputs.append({"assign_ids": assign_ids})
                pfeats_cls = self.bottelnecks[-1](torch.cat(pfeats_out, dim=1))
                losses = self.compute_losses_lvl(
                    pfeats_cls,
                    torch.cat(lvl_assign_ids, dim=1),
                    torch.cat(lvl_as_s, dim=1),
                    torch.cat(
                        [
                            det_priors[-(len_pfeats - i)]["pred_logits"]
                            for i in range(len_pfeats)
                        ],
                        dim=1,
                    ),
                )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        torch.cat(pfeats_out, dim=1),
                        torch.cat(lvl_assign_ids, dim=1),
                        torch.cat(lvl_as_s, dim=1),
                        torch.cat(
                            [
                                det_priors[-(len_pfeats - i)]["pred_logits"]
                                for i in range(len_pfeats)
                            ],
                            dim=1,
                        ),
                    )
                    losses.update(metric_losses)
                head_outputs["losses"].update(losses)
                head_outputs["assign_ids"] = assign_ids
                head_outputs["aux_outputs"] = head_aux_outputs
            else:
                assign_ids, as_s = self._id_assign_method(
                    det_priors[-1],
                    targets,
                    det_match_indices[-1],
                )
                pfeats_cls = self.bottelnecks[-1](pfeats_out[-1])
                losses = self.compute_losses_lvl(
                    pfeats_cls,
                    assign_ids,
                    as_s,
                    det_priors[-1]["pred_logits"],
                )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        pfeats_out[-1],
                        assign_ids,
                        as_s,
                        det_priors[-1]["pred_logits"],
                    )
                    losses.update(metric_losses)
                head_outputs["assign_ids"] = assign_ids
                # head_outputs["reid_feats"] = reid_feats
                head_outputs["losses"].update(losses)
        else:
            pfeats_cls = self.bottelnecks[-1](pfeats_out[-1])
            head_outputs["reid_feats"] = tF.normalize(pfeats_cls, dim=-1)
        return head_outputs


class DTransDecoderXHeadClean(DTransDecoderXHead):
    def split_id_asc_ptfeat_logits(self, det_outputs, targets, indices, det_feats):
        ids, as_s = self._id_assign_method(det_outputs, targets, indices)
        det_ids = ids
        bn_ids = []
        bn_asc = []
        bn_ptfeats = []
        bn_logits = []
        bn = ids.shape[0]
        num_pfeats = 0
        for bi in range(bn):
            mask = ids[bi] > -2
            bn_ids.append(ids[bi][mask])
            bn_asc.append(as_s[bi][mask] if as_s is not None else None)
            keep_pfeats = det_feats[bi][mask]
            keep_pfeats = keep_pfeats.view(-1, keep_pfeats.shape[-1])
            bn_ptfeats.append(keep_pfeats)
            num_pfeats += keep_pfeats.shape[0]
            bn_logits.append(det_outputs["pred_logits"][bi][mask])
        while num_pfeats < 2:
            # compatible with bn and dist
            bi = random.randint(0, bn - 1)
            li = random.randint(0, det_feats.shape[1] - 1)
            bi_ptfeats = bn_ptfeats[bi]
            bn_ptfeats[bi] = torch.cat([bi_ptfeats, det_feats[bi][li : li + 1]], dim=0)
            num_pfeats += 1
            bi_ids = bn_ids[bi]
            bn_ids[bi] = torch.cat([bi_ids, ids[bi][li : li + 1]], dim=0)
            bi_asc = bn_asc[bi]
            if bi_asc is not None:
                bn_asc[bi] = torch.cat([bi_asc, as_s[bi][li : li + 1]], dim=0)
            bi_logits = bn_logits[bi]
            bn_logits[bi] = torch.cat(
                [bi_logits, det_outputs["pred_logits"][bi][li : li + 1]], dim=0
            )

        return det_ids, bn_ids, bn_asc, bn_ptfeats, bn_logits

    def forward(
        self,
        det_outputs,
        targets,
        det_match_indices,
        *args,
        **kwargs,
    ):
        # NOTE aux share not kept
        det_priors, pfeats_out = self.get_pfeats(det_outputs)
        len_pfeats = len(pfeats_out)
        head_outputs = {}
        if self.training:
            head_outputs["losses"] = {}
            if len(self.loss_layers) > 1:
                head_aux_outputs = []
                lvl_assign_ids, lvl_as_s, lvl_pfeats, lvl_logits = [], [], [], []
                for i in range(len_pfeats):
                    (
                        assign_ids,
                        bn_ids,
                        bn_asc,
                        bn_ptfeats,
                        bn_logits,
                    ) = self.split_id_asc_ptfeat_logits(
                        det_priors[-(len_pfeats - i)],
                        targets,
                        det_match_indices[-(len_pfeats - i)],
                        pfeats_out[i],
                    )
                    lvl_assign_ids.append(bn_ids)
                    lvl_as_s.append(bn_asc)
                    lvl_pfeats.append(bn_ptfeats)
                    lvl_logits.append(bn_logits)
                    if i < len_pfeats - 1:
                        head_aux_outputs.append({"assign_ids": assign_ids})
                    else:
                        head_outputs["assign_ids"] = assign_ids
                aux_losses = {}
                for i in range(len_pfeats):
                    li_pfeats_mtc = torch.cat(lvl_pfeats[i], dim=0)
                    li_pfeats_cls = self.bottelnecks[i](li_pfeats_mtc)
                    li_assign_ids = torch.cat(lvl_assign_ids[i])
                    li_as_s = (
                        torch.cat(lvl_as_s[i]) if lvl_as_s[i][0] is not None else None
                    )
                    li_logits = torch.cat(lvl_logits[i])
                    losses = self.compute_losses_lvl(
                        li_pfeats_cls,
                        li_assign_ids,
                        li_as_s,
                        li_logits,
                        loss_layer_lvl=i,
                    )
                    if self.metric_loss is not None:
                        # TODO choose neck feat
                        metric_losses = self.compute_metric_losses(
                            li_pfeats_mtc,
                            li_assign_ids,
                            li_as_s,
                            li_logits,
                        )
                        losses.update(metric_losses)
                    if i < len_pfeats - 1:
                        for k, v in losses.items():
                            aux_losses[k + "_{}".format(i)] = v
                    else:
                        head_outputs["losses"].update(losses)
                head_outputs["losses"].update(aux_losses)
                head_outputs["aux_outputs"] = head_aux_outputs
            else:
                (
                    assign_ids,
                    bn_ids,
                    bn_asc,
                    bn_ptfeats,
                    bn_logits,
                ) = self.split_id_asc_ptfeat_logits(
                    det_priors[-1],
                    targets,
                    det_match_indices[-1],
                    pfeats_out[-1],
                )
                pfeats_mtc = torch.cat(bn_ptfeats, dim=0)
                pfeats_cls = self.bottelnecks[-1](pfeats_mtc)
                valid_ids = torch.cat(bn_ids)
                valid_as_s = torch.cat(bn_asc) if bn_asc[0] is not None else None
                valid_logits = torch.cat(bn_logits)
                losses = self.compute_losses_lvl(
                    pfeats_cls,
                    valid_ids,
                    valid_as_s,
                    valid_logits,
                )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        pfeats_mtc,
                        valid_ids,
                        valid_as_s,
                        valid_logits,
                    )
                    losses.update(metric_losses)
                head_outputs["assign_ids"] = assign_ids
                # head_outputs["reid_feats"] = reid_feats
                head_outputs["losses"].update(losses)
        else:
            pfeats_cls = self.bottelnecks[-1](pfeats_out[-1])
            head_outputs["reid_feats"] = tF.normalize(pfeats_cls, dim=-1)
        return head_outputs
