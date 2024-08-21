from psd2.modeling.meta_arch import GeneralizedRCNN
from psd2.structures.boxes import Boxes
from ..build import META_ARCH_REGISTRY
from ...backbone import build_backbone
import torch
from ...proposal_generator import build_proposal_generator
from psd2.modeling.reid_heads import build_rcnn_roi_reid_heads
import copy
from psd2.structures import ImageList, Instances
from psd2.utils.events import get_event_storage
from psd2.utils.visualizer import Visualizer, mlvl_pca_feat
from psd2.config import CfgNode as CN
from psd2.config import configurable


@META_ARCH_REGISTRY.register()
class RCNN_Baseline(GeneralizedRCNN):
    @configurable
    def __init__(self, *, det_loss_weights, **kwargs):
        super(RCNN_Baseline, self).__init__(**kwargs)
        self.det_loss_weights = det_loss_weights

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        det_loss_cfg = cfg.DETECTOR.LOSS.LOSS_WEIGHTS
        det_loss_weights = {
            "loss_rpn_cls": det_loss_cfg.RPN_CLS,
            "loss_rpn_loc": det_loss_cfg.RPN_LOC,
            "loss_cls": det_loss_cfg.CLS,
            "loss_box_reg": det_loss_cfg.BOX_REG,
        }
        head_cfg = CN()
        head_cfg.REID_HEAD = copy.deepcopy(cfg.REID_HEAD)
        head_cfg.MODEL = copy.deepcopy(cfg.DETECTOR.MODEL)
        head_cfg.defrost()
        head_cfg.MODEL.RESNETS = copy.deepcopy(cfg.MODEL.RESNETS)
        head_cfg.TEST = copy.deepcopy(cfg.TEST)
        head_cfg.freeze()
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(
                cfg, backbone.output_shape()
            ),
            "roi_heads": build_rcnn_roi_reid_heads(head_cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "det_loss_weights": det_loss_weights,
        }

    def inf_query(self, input_list):
        inputs = [qd["query"] for qd in input_list]
        images, gt_instances = self.preprocess_input(inputs)
        features = self.backbone(images.tensor)
        proposal_boxes = [inst.gt_boxes for inst in gt_instances]
        res4_features = self.roi_heads.pooler(
            [features[f] for f in self.roi_heads.in_features], proposal_boxes
        )
        res4_embs = res4_features.mean(dim=[2, 3])
        res5_features = self.roi_heads.res5(res4_features)
        res5_embs = res5_features.mean(dim=[2, 3])
        pfeats = self.roi_heads.reid_feat({"res4": res4_embs, "res5": res5_embs})
        result_list = input_list.copy()
        for bi, feat in enumerate(pfeats):
            result_list[bi]["query"]["feat"] = feat
        return result_list

    def forward(self, input_list):
        if "query" in input_list[0]:
            return self.inf_query(input_list)
        images, gt_instances = self.preprocess_input(input_list)

        features = self.backbone(images.tensor)
        if self.training:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
            outputs, det_reid_losses = self.roi_heads(features, proposals, gt_instances)
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(
                        (images, gt_instances),
                        features[list(features.keys())[-1]],
                        outputs,
                    )
            losses = {}
            losses.update(det_reid_losses)
            losses.update(proposal_losses)
            for lk, lv in self.det_loss_weights.items():
                losses[lk] *= lv
            return losses
        else:
            proposals, _ = self.proposal_generator(images, features, None)
            outputs, _ = self.roi_heads(features, proposals, None)
            for i, (pred_boxes, im_s, o_im_s) in enumerate(
                zip(
                    outputs["pred_boxes"],
                    images.image_sizes,
                    [inst._org_hw for inst in gt_instances],
                )
            ):
                org_boxes = _resize_boxes(pred_boxes, im_s, o_im_s)
                outputs["pred_boxes"][i] = org_boxes
        return outputs

    def preprocess_input(self, input_list):
        images = []
        gt_instances = []
        for input_dict in input_list:
            inst_img = Instances((input_dict["height"], input_dict["width"]))
            inst_img.gt_boxes = Boxes(
                torch.tensor(
                    input_dict["boxes"],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            inst_img._file_name = input_dict["file_name"]
            inst_img._image_id = input_dict["image_id"]
            images.append(input_dict["image"].to(self.device))
            org_pids = torch.tensor(input_dict["ids"], device=self.device).clone()
            org_pids[org_pids > -1] += 2  # labeled people pid +=2
            org_pids[org_pids == -1] = 0  # unlabeled people pid =0
            inst_img.gt_classes = org_pids.long()
            inst_img._org_hw = (input_dict["org_height"], input_dict["org_width"])
            inst_img.org_boxes = Boxes(torch.tensor(input_dict["org_boxes"]))
            gt_instances.append(inst_img.to(self.device))
        return (
            ImageList.from_tensors(images, self.backbone.size_divisibility),
            gt_instances,
        )

    @torch.no_grad()
    def visualize_training(self, batched_inputs, featmap, batched_dets):
        """
        Args:
            batched_inputs:
                [imgs nested tensor, img_gt_instances]
            featmap: a batched images feature map tensor
            featmap: (multi level) image feature map(s)
            batched_dets:
                {
                    "pred_logits": batched person scores B x N x 1,
                    "pred_boxes": batched xyxy bboxes in [0,1],
                    "reid_feat": reid feature,
                    "assign_ids": ids assigned to each reid feature
                    "aux_outputs": optional,
                ]
            vals: scalars to be visualized
            thred: cls score threshold
        """
        import torchvision.transforms.functional as tvF
        import numpy as np

        COLORS = ["r", "g", "b", "y", "c", "m"]
        T_COLORS_BG = {
            "r": "white",
            "g": "white",
            "b": "white",
            "y": "black",
            "c": "black",
            "m": "white",
        }

        def score_splits(score, threds):
            splits = []
            for i, v in enumerate(threds):
                if score >= threds[i]:
                    splits.append(i)
            return splits

        threds = [0.5]
        storage = get_event_storage()
        trans_t2img_t = lambda t: t.detach().cpu() * self.pixel_std.cpu().view(
            -1, 1, 1
        ) + self.pixel_mean.cpu().view(-1, 1, 1)
        img_t2rgb = lambda t: (t.permute(1, 2, 0) * 255).numpy()
        samples = batched_inputs[0]
        annos = []
        for inst in batched_inputs[1]:
            trans_id = inst.gt_classes
            trans_id[trans_id == 0] = -1
            trans_id[trans_id > 1] -= 2
            annos.append(
                {
                    "file_name": inst._file_name,
                    "image_id": inst._image_id,
                    "boxes": inst.gt_boxes.tensor,  # xyxy abs
                    "ids": trans_id,
                }
            )
        bs = len(annos)
        level_pcas = mlvl_pca_feat(featmap[None])
        for bi in range(bs):
            img_norm_t = samples.tensor[bi].cpu()
            img_t = trans_t2img_t(img_norm_t)
            img_rgb = img_t2rgb(img_t)
            visualize_org = Visualizer(img_rgb.copy())
            boxes = annos[bi]["boxes"].cpu()
            ids = annos[bi]["ids"].tolist()
            for i in range(boxes.shape[0]):
                visualize_org.draw_box(boxes[i])
                id_pos = boxes[i, :2]
                visualize_org.draw_text(
                    str(ids[i]), id_pos, horizontal_alignment="left", color="w"
                )

            rgb_org_ann = visualize_org.get_output().get_image()
            t_org_ann = tvF.to_tensor(rgb_org_ann)
            storage.put_image("img_{}/gt".format(bi), t_org_ann)

            visualize_runs = [Visualizer(img_rgb.copy()) for i in range(len(threds))]
            boxes = batched_dets["pred_boxes"][bi].detach().cpu()
            # TODO check if topk needed
            scores = (
                batched_dets["pred_scores"][bi]
                .detach()
                .cpu()
                .squeeze(1)
                .numpy()
                .tolist()
            )
            for i, score in enumerate(scores):
                splits = score_splits(score, threds)
                if len(splits) == 0:
                    continue
                b_clr = COLORS[i % len(COLORS)]
                t_clr = T_COLORS_BG[b_clr]
                for split in splits:
                    visualize_runs[split].draw_box(boxes[i], edge_color=b_clr)
                    visualize_runs[split].draw_text(
                        "%.2f" % score,
                        boxes[i][2:],
                        horizontal_alignment="right",
                        color=t_clr,
                        bg_color=b_clr,
                    )
                    visualize_runs[split].draw_text(
                        str(batched_dets["assign_ids"][bi][i].item()),
                        boxes[i][:2],
                        horizontal_alignment="left",
                        color=t_clr,
                        bg_color=b_clr,
                    )
            rgb_run_vis = [vis.get_output().get_image() for vis in visualize_runs]
            rgb_run_vis = np.concatenate(rgb_run_vis, axis=1)
            t_run_vis = tvF.to_tensor(rgb_run_vis)
            storage.put_image("img_{}/det".format(bi), t_run_vis)
            max_h, w = 0, 0
            level_pca_feats = []
            for feat in level_pcas:
                bi_pca = feat[bi]
                level_pca_feats.append(bi_pca)
                if bi_pca.shape[-2] > max_h:
                    max_h = bi_pca.shape[-2]
                w += bi_pca.shape[-1]
            bg = torch.zeros(3, max_h, w, dtype=torch.float32)
            next_w = 0
            for feat in level_pcas:
                bg[
                    :, : feat[bi].shape[-2], next_w : next_w + feat[bi].shape[-1]
                ] = feat[bi]
                next_w += feat[bi].shape[-1]
            storage.put_image("img_{}/feat".format(bi), bg / 255)


def _resize_boxes(boxes, original_size, new_size):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
