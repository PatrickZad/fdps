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
class SparseRCNN_PS_SS(SparseRCNN_PS_DC):
    """
    Implement SparseRCNN for Shared Sparse Feature Person Search
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
            return input_list
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

        pred_psfeats = outputs_pfeats  # D x B x Nq x 256

        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "reid_feats": pred_psfeats[-1],
        }  # xyxy boxes in aug
        output["aux_outputs"] = [
            {"pred_logits": a, "pred_boxes": b, "reid_feats": c}
            for a, b, c in zip(
                outputs_class[:-1], outputs_coord[:-1], pred_psfeats[:-1]
            )
        ]

        if self.training:
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
            inter_outs = output.pop("aux_outputs")
            feat_lvl = self.cfg.MODEL.SEARCH.PERSON_FEAT.FEAT_BASE_LVL_IDX
            if feat_lvl < len(inter_outs):
                output["reid_feats"] = inter_outs[feat_lvl]["reid_feats"]
            return output
