import itertools
import torch
from torch import nn
from ..build import META_ARCH_REGISTRY
from psd2.structures.boxes import box_cxcywh_to_xyxy
from .sparse_ps_baseline import SparsePS_Baseline
from psd2.utils.events import get_event_storage
from psd2.utils.visualizer import domain_hist_img
import numpy as np
from psd2.structures.nested_tensor import NestedTensor
from psd2.layers import _get_clones
import logging

logger = logging.getLogger("psd2")


@META_ARCH_REGISTRY.register()
class Incmt_SparsePS_Baseline(SparsePS_Baseline):
    """
    Implement SparseRCNN for Decoupled Person Search
    """

    def _local_init(self, *args, **kws):
        super()._local_init(*args, **kws)

        # Build Proposals.
        self.init_proposal_features_novel = nn.Embedding(
            self.num_proposals, self.hidden_dim
        )
        self.init_proposal_boxes_novel = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes_novel.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes_novel.weight[:, 2:], 1.0)

        self.train_with_inc_queries = self.cfg.DETECTOR.MODEL.TRAIN_WITH_INC_QUERY
        self.novel_query_tune = self.cfg.DETECTOR.MODEL.NOVEL_QUERY_TUNE

    def load_state_dict(self, *args, **kws):
        output = super().load_state_dict(*args, **kws)  #
        for parameter in itertools.chain(
            self.init_proposal_features.parameters(),
            self.init_proposal_boxes.parameters(),
        ):
            parameter.requires_grad_(False)
        if self.novel_query_tune:
            with torch.nograd():
                self.init_proposal_features_novel.weight.copy_(
                    self.init_proposal_features.weight.clone().detach()
                )
                self.init_proposal_boxes_novel.weight.copy_(
                    self.init_proposal_boxes.weight.clone().detach()
                )
        # "tail",
        for param_k in output.missing_keys:
            if "ulb_layer.queue" in param_k:
                for oim_layer in self.reid_head.loss_layers:
                    oim_layer.ulb_layer.tail = torch.tensor([0], device=self.device)
                break
        return output

    def get_det_pred(self, img_batch_tensor, img_whwh):
        # Feature Extraction.
        src = self.backbone(img_batch_tensor)
        res_feats = src["bk_feats"]
        lateral_feats = list()
        out_feats = src["out"]
        features = list()
        for f in self.in_features:
            feature = out_feats[f]
            features.append(feature)
            lateral_feats.append(src["lateral"][f])
        if self.train_with_inc_queries or not self.training:
            proposal_boxes = torch.cat(
                [
                    self.init_proposal_boxes.weight.clone(),
                    self.init_proposal_boxes_novel.weight.clone(),
                ],
                dim=0,
            )
            proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
            proposal_boxes = (
                proposal_boxes[None] * img_whwh[:, None, :]
            )  # xyxy_abs in aug
            proposal_feats = torch.cat(
                [
                    self.init_proposal_features.weight,
                    self.init_proposal_features_novel.weight,
                ],
                dim=0,
            )

        else:
            # Prepare Proposals.
            proposal_boxes = self.init_proposal_boxes_novel.weight.clone()
            proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
            proposal_boxes = (
                proposal_boxes[None] * img_whwh[:, None, :]
            )  # xyxy_abs in aug
            proposal_feats = self.init_proposal_features_novel.weight

        # Prediction.
        return (
            res_feats,
            lateral_feats,
            features,
            self.det_head(features, proposal_boxes, proposal_feats),
        )

    def forward(self, input_list):
        output = super().forward(input_list)
        if not self.training and not "query" in input_list[0]:
            num_query_per_domain = self.init_proposal_boxes.weight.shape[0]
            output["pred_scores_domain"] = torch.split(
                output["pred_scores"], num_query_per_domain, dim=1
            )
        return output

    @torch.no_grad()
    def visualize_training(self, batched_inputs, featmap, batched_dets):
        super().visualize_training(batched_inputs, featmap, batched_dets)
        storage = get_event_storage()
        num_query_per_domain = self.init_proposal_boxes.weight.shape[0]
        bs = batched_dets["pred_logits"].shape[0]
        for bi in range(bs):
            scores = (
                batched_dets["pred_logits"][bi]
                .detach()
                .cpu()
                .sigmoid()
                .squeeze(1)
                .numpy()
            )
            domain_scores = np.split(scores, scores.shape[0] // num_query_per_domain)
            hist_rgb = domain_hist_img(domain_scores, 10)
            hist_img_t = (
                torch.tensor(hist_rgb, dtype=torch.float32).permute(2, 0, 1) / 255
            )
            storage.put_image("img_{}/d_scores".format(bi), hist_img_t)


@META_ARCH_REGISTRY.register()
class Transfer_SparsePS_Baseline(SparsePS_Baseline):
    def load_state_dict(self, *args, **kws):
        output = super().load_state_dict(*args, **kws)
        for param_k in output.missing_keys:
            if "ulb_layer.queue" in param_k:
                for oim_layer in self.reid_head.loss_layers:
                    oim_layer.ulb_layer.tail = torch.tensor([0], device=self.device)
                break
        if (
            self.cfg.REID_HEAD.NAME.endswith("_side")
            and self.cfg.REID_HEAD.PRETAIN_SIDE_NET
        ):

            def valid_side_name(name, param_dict):
                sp_names = name.split(".")
                if (
                    sp_names[0].endswith("_side")
                    and ".".join([sp_names[0][:-5]] + sp_names[1:]) in param_dict
                ):
                    return ".".join([sp_names[0][:-5]] + sp_names[1:])
                else:
                    return None

            param_dict = self.reid_head.state_dict()
            for name, param in list(self.reid_head.named_parameters()) + list(
                self.reid_head.named_buffers()
            ):

                valide_name = valid_side_name(name, param_dict)
                if valide_name:
                    full_name = "reid_head." + name
                    if full_name in output.missing_keys:
                        with torch.no_grad():
                            param.copy_(param_dict[valide_name])
                        output.missing_keys.remove(full_name)
                        logger.info("Removed " + full_name + " from missing_keys")
                    else:
                        # NOTE only copy missing_keys
                        logger.info(full_name + " is not in missing_keys")

        return output

    """
    def forward(self, input_list):
        output = super().forward(input_list)
        if not self.training and not "query" in input_list[0]:
            num_query_per_domain = self.init_proposal_boxes.weight.shape[0]
            output["pred_scores_domain"] = torch.split(
                output["pred_scores"], num_query_per_domain, dim=1
            )
        return output
    """

    @torch.no_grad()
    def visualize_training(self, batched_inputs, featmap, batched_dets):
        super().visualize_training(batched_inputs, featmap, batched_dets)
        storage = get_event_storage()
        num_query_per_domain = self.init_proposal_boxes.weight.shape[0]
        bs = batched_dets["pred_logits"].shape[0]
        for bi in range(bs):
            scores = (
                batched_dets["pred_logits"][bi]
                .detach()
                .cpu()
                .sigmoid()
                .squeeze(1)
                .numpy()
            )
            domain_scores = np.split(scores, scores.shape[0] // num_query_per_domain)
            hist_rgb = domain_hist_img(domain_scores, 10)
            hist_img_t = (
                torch.tensor(hist_rgb, dtype=torch.float32).permute(2, 0, 1) / 255
            )
            storage.put_image("img_{}/d_scores".format(bi), hist_img_t)


@META_ARCH_REGISTRY.register()
class SparsePS_Baseline_GTDet(SparsePS_Baseline):
    def forward(self, input_list):
        assert not self.training
        if "query" in input_list[0]:
            return self.inf_query(input_list)
        input_batches = self.preprocess_input(input_list)
        img_nested_tensor: NestedTensor = input_batches[0]
        # Feature Extraction.
        src = self.backbone(img_nested_tensor.tensors)
        lateral_feats = list()
        out_feats = src["out"]
        features = list()
        for f in self.in_features:
            feature = out_feats[f]
            features.append(feature)
            lateral_feats.append(src["lateral"][f])
        targets = self.prepare_targets(input_batches)
        if "ms" in self.cfg.REID_HEAD.NAME:
            reid_feats = self.reid_head(
                lateral_feats, None, targets, det_match_indices=None
            )["reid_feats"]
        else:
            reid_feats = self.reid_head(
                features, None, targets, det_match_indices=None
            )["reid_feats"]
        gt_scores = [
            torch.ones(
                bs.shape[0], dtype=reid_feats[0].dtype, device=self.device
            ).unsqueeze(1)
            for bs in input_batches[3]
        ]
        output = {
            "pred_scores": gt_scores,
            "pred_boxes": input_batches[3],
        }
        aug_hws = output["pred_boxes"][0].new_tensor(input_batches[5])
        org_hws = output["pred_boxes"][0].new_tensor(input_batches[6])  # B x 2
        org_whwh = torch.stack([org_hws[:, 1], org_hws[:, 0]] * 2, dim=1)  # B x 4
        aug_whwh = torch.stack([aug_hws[:, 1], aug_hws[:, 0]] * 2, dim=1)  # B x 4
        sf = org_whwh / aug_whwh
        for bi in range(sf.shape[0]):
            output["pred_boxes"][bi] *= sf[bi].unsqueeze(0)
        output["reid_feats"] = reid_feats
        return output


@META_ARCH_REGISTRY.register()
class PTKP_SparsePS_Baseline_ExpFree(SparsePS_Baseline):
    def _local_init(self, *args, **kws):
        super()._local_init(*args, **kws)
        domains = self.cfg.DATASETS.TRAIN + self.cfg.DATASETS.INCMT.TRAIN
        self.num_domains = len(domains)
        self.domain_id_offsets = self.cfg.DATASETS.INCMT.ID_OFFSETS
        self.domain_bns = _get_clones(self.reid_head.feat_bn)
        self.register_buffer("current_domain", torch.tensor([0]))

    def forward(self, input_list, domain_models):
        pass
