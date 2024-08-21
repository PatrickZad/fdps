import itertools
from .base_reid_head import ReidHeadBase
import torch.nn as nn
import torch
import random
import torch.nn.functional as tF
from .dt_decoder_head import CompBN1D


class BnneckHead(ReidHeadBase):
    def __init__(self, head_cfg):
        super().__init__(head_cfg)
        self.bn_necks = nn.ModuleList(
            [CompBN1D(self.pfeat_dim) for _ in range(len(self.loss_layers))]
        )

    def split_id_asc_ptfeat_logits(self, ids, as_s, det_outputs, det_feats):
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

        return bn_ids, bn_asc, bn_ptfeats, bn_logits

    def forward(self, pfeats_out, det_outputs, targets, det_match_indices, *args, **kw):
        # TODO cleaner bn
        head_outputs = {}
        len_pfeats = len(pfeats_out)
        if self.training:
            head_outputs["losses"] = {}
            n_losses = len(self.loss_layers)
            if n_losses > 1 or self.shared_aux:
                # TODO compute only with ids
                assert "aux_outputs" in det_outputs
                head_aux_outputs = []
                inter_outs = det_outputs["aux_outputs"] + [
                    {
                        "pred_logits": det_outputs["pred_logits"],
                        "pred_boxes": det_outputs["pred_boxes"],
                    }
                ]
                lvl_assign_ids, lvl_as_s, lvl_pfeats, lvl_logits = [], [], [], []
                loss_i_offset = len_pfeats - n_losses
                for i in range(n_losses):
                    assign_ids, as_s = self._id_assign_method(
                        inter_outs[i + loss_i_offset],
                        targets,
                        det_match_indices[i + loss_i_offset],
                    )
                    (
                        bn_ids,
                        bn_asc,
                        bn_ptfeats,
                        bn_logits,
                    ) = self.split_id_asc_ptfeat_logits(
                        assign_ids,
                        as_s,
                        inter_outs[i + loss_i_offset],
                        pfeats_out[i + loss_i_offset],
                    )
                    lvl_assign_ids.append(bn_ids)
                    lvl_as_s.append(bn_asc)
                    lvl_pfeats.append(bn_ptfeats)
                    lvl_logits.append(bn_logits)
                    if i < n_losses - 1:
                        head_aux_outputs.append({"assign_ids": assign_ids})
                    else:
                        head_outputs["assign_ids"] = assign_ids
                head_outputs["aux_outputs"] = head_aux_outputs
                if not self.shared_aux:
                    aux_losses = {}
                    for i in range(n_losses):
                        li_pfeats_mtc = torch.cat(lvl_pfeats[i], dim=0)
                        li_pfeats_cls = self.bn_necks[i](li_pfeats_mtc)
                        li_assign_ids = torch.cat(lvl_assign_ids[i])
                        li_as_s = (
                            torch.cat(lvl_as_s[i])
                            if lvl_as_s[i][0] is not None
                            else None
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
                        if i < n_losses - 1:
                            for k, v in losses.items():
                                aux_losses[k + "_{}".format(i)] = v
                        else:
                            head_outputs["losses"].update(losses)
                    head_outputs["losses"].update(aux_losses)
                else:
                    all_assign_ids = torch.cat(list(itertools.chain(*lvl_assign_ids)))
                    all_as_s = (
                        torch.cat(list(itertools.chain(*lvl_as_s)))
                        if lvl_as_s[i][0] is not None
                        else None
                    )
                    all_pfeats_mtc = torch.cat(
                        list(itertools.chain(*lvl_pfeats)), dim=0
                    )
                    all_logits = torch.cat(list(itertools.chain(*lvl_logits)))
                    all_pfeats_cls = self.bn_necks[-1](all_pfeats_mtc)
                    losses = self.compute_losses_lvl(
                        all_pfeats_cls,
                        all_assign_ids,
                        all_as_s,
                        all_logits,
                    )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        all_pfeats_cls,
                        all_assign_ids,
                        all_as_s,
                        all_logits,
                    )
                    losses.update(metric_losses)
                head_outputs["losses"].update(losses)

            else:
                assign_ids, as_s = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )
                (
                    bn_ids,
                    bn_asc,
                    bn_ptfeats,
                    bn_logits,
                ) = self.split_id_asc_ptfeat_logits(
                    assign_ids,
                    as_s,
                    det_outputs,
                    pfeats_out[-1],
                )
                pfeats_mtc = torch.cat(bn_ptfeats, dim=0)
                pfeats_cls = self.bn_necks[-1](pfeats_mtc)
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
            pfeats_cls = self.bn_necks[-1](pfeats_out[-1])
            head_outputs["reid_feats"] = tF.normalize(pfeats_cls, dim=-1)
        return head_outputs
