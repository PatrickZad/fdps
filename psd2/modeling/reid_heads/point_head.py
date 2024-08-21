import random
import torch
from .base_reid_head import ReidHeadBase
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as tF


class PointHead(ReidHeadBase):
    def __init__(self, head_cfg):
        super().__init__(head_cfg)
        self.train_feat_at = head_cfg.PERSON_FEATURE.TRAIN_AT
        self.inf_feat_at = head_cfg.PERSON_FEATURE.INF_AT
        self.feat_bn = nn.BatchNorm1d(self.pfeat_dim)
        init.normal_(self.feat_bn.weight, std=0.01)
        init.constant_(self.feat_bn.bias, 0)

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

    def get_pfeats(self, b_pt_feats):
        if isinstance(b_pt_feats, torch.Tensor):  # b x l x c
            bn_feats = self.feat_bn(b_pt_feats.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            num_splits = [feats.shape[0] for feats in b_pt_feats]
            cat_feats = torch.cat(b_pt_feats, dim=0)  # ni x c
            cat_bn_feats = self.feat_bn(cat_feats)
            bn_feats = torch.split(cat_bn_feats, num_splits)
        return b_pt_feats, bn_feats

    def forward(self, bk_feats, det_outputs, targets, det_match_indices, *args, **kw):
        # TODO append GT
        head_outputs = {}
        det_feats = bk_feats[-1].permute(0, 2, 3, 1).flatten(1, 2)
        if self.training:
            # NOTE update to only involve boxes with valid ids
            head_outputs["losses"] = {}
            (
                assign_ids,
                bn_ids,
                bn_asc,
                bn_ptfeats,
                bn_logits,
            ) = self.split_id_asc_ptfeat_logits(
                det_outputs, targets, det_match_indices[-1], det_feats
            )
            # compute only with valid ids
            bn_reid_feats_metric, bn_reid_feats_cls = self.get_pfeats(bn_ptfeats)
            losses = self.compute_losses_lvl(
                torch.cat(bn_reid_feats_cls, dim=0),
                torch.cat(bn_ids, dim=0),
                torch.cat(bn_asc, dim=0) if bn_asc[0] is not None else None,
                torch.cat(bn_logits, dim=0),
            )
            if self.metric_loss is not None:
                metric_losses = self.compute_metric_losses(
                    torch.cat(bn_reid_feats_metric, dim=0),
                    torch.cat(bn_ids, dim=0),
                    torch.cat(bn_asc, dim=0) if bn_asc[0] is not None else None,
                    torch.cat(bn_logits, dim=0),
                )
                losses.update(metric_losses)
            head_outputs["assign_ids"] = assign_ids
            # head_outputs["reid_feats"] = reid_feats
            head_outputs["losses"].update(losses)

        else:
            # gallery feat
            _, det_pfeats = self.get_pfeats(det_feats)
            head_outputs["reid_feats"] = tF.normalize(det_pfeats, dim=-1)
        return head_outputs
