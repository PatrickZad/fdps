#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
from ..build import META_ARCH_REGISTRY
from psd2.structures import Boxes
from psd2.structures.boxes import box_cxcywh_to_xyxy
from psd2.structures.nested_tensor import NestedTensor
from .sparse_rcnn_dcps import SparseRCNN_PS_DC
from psd2.structures import Boxes
from psd2.utils.events import get_event_storage


@META_ARCH_REGISTRY.register()
class SparseRCNN_PS_SP(SparseRCNN_PS_DC):
    """
    Implement SparseRCNN for Shared ROI Feature Person Search
    """

    def forward(self, input_list):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        if "query" in input_list[0]:
            return self.inf_query(input_list)
        input_batches = self.preprocess_input(input_list)
        img_nested_tensor: NestedTensor = input_batches[0]
        images_whwh = img_nested_tensor.tensors.new_tensor(
            [hw[::-1] * 2 for hw in img_nested_tensor.image_sizes]
        )
        targets = self.prepare_targets(input_batches)
        features, (outputs_class, outputs_coord, outputs_pfeats) = self.get_det_pred(
            img_nested_tensor.tensors, images_whwh
        )
        # scale xyxy abs coords to [0,1] in aug_size
        aug_hw_bs = images_whwh.new_tensor(input_batches[5])  # B x 2
        aug_whwh_bs = torch.stack(
            [aug_hw_bs[:, 1], aug_hw_bs[:, 0]] * 2, dim=1
        )  # B x 4
        box_pooler = self.head.box_pooler
        nd, nb, nq, lb = outputs_coord.shape
        roi_boxes = outputs_coord.permute(1, 0, 2, 3).reshape(
            nb, -1, lb
        )  # B x (D x Nq) x 4

        if self.training and self.append_gt:
            boxes_list = [
                Boxes(input_batches[3][bi]) for bi in range(nb)
            ]  # append gt boxes
            rois_feats = box_pooler(features, boxes_list)  # (Bx[g]) x 256 x 7 x 7
            rois_feats = rois_feats.flatten(start_dim=1)
            fl_outputs_pfeats = outputs_pfeats.flatten(3).permute(
                1, 0, 2, 3
            )  #  B x D x Nq x (256*7*7)
            rois_feats = torch.cat(
                [rois_feats, fl_outputs_pfeats.view(-1, fl_outputs_pfeats.shape[-1])],
                dim=0,
            )
            psfeats = self.pfeat_head(rois_feats)  # (Bx[g]+B x D x Nq) x 256
            num_gts = [input_batches[3][bi].shape[0] for bi in range(nb)]
            num_qs = [nd * nq for _ in range(nb)]
            split_sizes = num_gts + num_qs
            psfeats_splits = torch.split(psfeats, split_sizes)
            pred_psfeats = (
                torch.cat(psfeats_splits[nb:], dim=0)
                .view(nb, nd, nq, -1)
                .permute(1, 0, 2, 3)
            )  # D x B x Nq x 256
            gt_psfeats = torch.cat(psfeats_splits[:nb], dim=0)
            ids_list = []
            for ids in zip(input_batches[4]):
                ids_list.append(
                    torch.tensor(ids, dtype=torch.int, device=self.device).squeeze(0)
                )
            gt_ids = torch.cat(ids_list, dim=-1).view(-1)
        else:
            fl_outputs_pfeats = outputs_pfeats.flatten(3).permute(
                1, 0, 2, 3
            )  #  B x D x Nq x (256*7*7)
            pred_psfeats = self.pfeat_head(
                fl_outputs_pfeats.view(-1, fl_outputs_pfeats.shape[-1])
            )  # (BxDxNq) x 256
            pred_psfeats = pred_psfeats.view(nb, nd, nq, -1).permute(
                1, 0, 2, 3
            )  # D x B x Nq x 256
        """outputs_coord = (
            outputs_coord / aug_whwh_bs[None, :, None, :]
        )  # D x B x Nq x 4 xyxy [0,1] in aug_size"""

        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "reid_feats": pred_psfeats[-1],
        }  # xyxy boxes in aug [0,1]

        if self.training:
            if self.deep_supervision:
                output["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b, "reid_feats": c}
                    for a, b, c in zip(
                        outputs_class[:-1], outputs_coord[:-1], pred_psfeats[:-1]
                    )
                ]
            if self.append_gt:
                loss_dict = self.criterion(
                    output, targets, {"reid_feats": gt_psfeats, "ids": gt_ids}
                )
            else:
                loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            if get_event_storage().iter % self.vis_period == 0:
                self.visualize_training(
                    input_batches,
                    features,
                    output,
                    list(gt_psfeats.split(num_gts)),
                    ids_list,
                )
            return loss_dict

        else:
            aug_hws = output["pred_boxes"].new_tensor(input_batches[5])
            org_hws = output["pred_boxes"].new_tensor(input_batches[6])  # B x 2
            logits = output.pop("pred_logits", None)
            output["pred_scores"] = logits.sigmoid()
            org_whwh = torch.stack([org_hws[:, 1], org_hws[:, 0]] * 2, dim=1)  # B x 4
            aug_whwh = torch.stack([aug_hws[:, 1], aug_hws[:, 0]] * 2, dim=1)  # B x 4
            sf = org_whwh / aug_whwh
            # back to org abs
            output["pred_boxes"] *= sf.unsqueeze(1)  # B x 1 x 4

            inter_outs = output.pop("aux_outputs", None)
            return output
