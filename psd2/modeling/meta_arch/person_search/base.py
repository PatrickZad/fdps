import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...backbone import build_backbone
from ..build import META_ARCH_REGISTRY
from psd2.structures.nested_tensor import nested_collate_fn, NestedTensor
from psd2.utils.events import get_event_storage
from psd2.utils.visualizer import pca_feat, Visualizer
from psd2.structures.boxes import box_cxcywh_to_xyxy
import torchvision.transforms.functional as tvF
from PIL import Image

COLORS = ["r", "g", "b", "y", "c", "m"]
T_COLORS_BG = {
    "r": "white",
    "g": "white",
    "b": "white",
    "y": "black",
    "c": "black",
    "m": "white",
}


@META_ARCH_REGISTRY.register()
class SearchBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.vis_period = cfg.VIS_PERIOD

    @property
    def device(self):
        return self.pixel_mean.device

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
            cxcy = (xyxy_box_t[:, :2] + xyxy_box_t[:, 2:]) / 2
            wh = xyxy_box_t[:, 2:] - xyxy_box_t[:, :2]
            img_boxes.append(torch.cat([cxcy, wh], dim=-1))
            img_pids.append(input_dict["ids"])
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

    def forward(self, input_list):
        if "query" in input_list[0].keys():
            return input_list
        else:
            batched_input = self.preprocess_input(input_list)
            return self.run_iter(batched_input)

    def get_prediction(self, batched_input):
        """
        Return cls_scores, bboxes and reid features
        """
        raise NotImplementedError

    def losses(self, output_dict):
        """
        Args:
            cls_scores, bboxes and reid features
        """
        raise NotImplementedError

    def run_iter(batched_input):
        raise NotImplementedError

    def visualize_training(
        self, batched_inputs, featmap, batched_dets, vals={}, is_ccwh=True
    ):
        """
        Args:
            batched_inputs:
                [imgs nested tensor, imgs paths, imgs ids, imgs bboxes, imgs person ids]
            featmap: a batched images feature map tensor
            featmap: (multi level) image feature map(s)
            batched_dets:
                {
                    "pred_logits": batched person scores B x N x 1,
                    "pred_boxes": batched xyxy bboxes in aug,
                    "pred_ref_pts": batched decoder query refrence points in [0,1],
                    "reid_feat": reid feature,
                    "assign_ids": ids assigned to each reid feature
                    "aux_outputs": optional,
                    "enc_outputs": optional
                ]
            vals: scalars to be visualized
            thred: cls score threshold
        """

        def score_split(score, threds):
            for i, v in enumerate(threds[:-1]):
                if score >= threds[i] and score < threds[i + 1]:
                    return i

        threds = [0, 0.2, 0.5, 0.7, 1]
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
                    "boxes": bboxes,
                    "ids": ids.cpu().numpy().tolist(),
                }
            )
        bs = len(annos)
        level_pcas = []
        for feat in featmap:
            feat_pca = pca_feat(feat.detach())
            level_pcas.append(feat_pca)
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
            visualize_run_ptses = [
                Visualizer(img_rgb.copy()) for i in range(len(threds) - 1)
            ]
            img_h, img_w = img_norm_t.shape[-2:]
            boxes = batched_dets["pred_boxes"][bi].detach().cpu()
            # boxes = boxes * boxes.new_tensor(((img_w, img_h) * 2,))
            if is_ccwh:
                boxes = box_cxcywh_to_xyxy(boxes)
            ref_pts = batched_dets["pred_ref_pts"][bi].detach().cpu()
            ref_pts = ref_pts  # * ref_pts.new_tensor(((img_w, img_h),))
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
                    str(batched_dets["assign_ids"][bi][i].item()),
                    boxes[i][:2],
                    horizontal_alignment="left",
                    color=t_clr,
                    bg_color=b_clr,
                )
                visualize_run_ptses[split].draw_circle(
                    ref_pts[i].tolist(), color=b_clr, radius=5
                )
            rgb_run_vis = [vis.get_output().get_image() for vis in visualize_runs]
            rgb_run_vis = np.concatenate(rgb_run_vis, axis=1)
            t_run_vis = tvF.to_tensor(rgb_run_vis)
            storage.put_image("img_{}/det".format(bi), t_run_vis)
            rgb_run_pts = [vis.get_output().get_image() for vis in visualize_run_ptses]
            rgb_run_pts = np.concatenate(rgb_run_pts, axis=1)
            t_run_pts = tvF.to_tensor(rgb_run_pts)
            storage.put_image("img_{}/det_rpts".format(bi), t_run_pts)
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
        # scalar vis
        for k, v in vals.items():
            storage.put_scalar(k, v)

    def visualize_inference(
        self,
        batched_inputs,
        batched_dets,
        thred=0.5,
    ):
        """
        Args:
            batched_inputs:
                [imgs nested tensor, imgs paths, imgs ids, imgs bboxes, imgs person ids,
                img_aug_hws, img_org_hws, img_org_boxes,]
            batched_dets:
                {
                    "pred_scores": batched person scores B x N x 1,
                    "pred_boxes": batched xyxy bboxes in org,
                    "pred_feats": batch reid feats
                }
            thred: cls score threshold
        """
        from psd2.structures.boxes import box_cxcywh_to_xyxy

        bs = batched_dets["pred_scores"].shape[0]
        for bi in range(bs):
            img_path = batched_inputs[1][bi]
            img_id = batched_inputs[2][bi]
            img = Image.open(img_path)
            vis_org = Visualizer(img.copy())
            vis_det = Visualizer(img.copy())
            org_boxes = batched_inputs[7][bi]
            org_ids = batched_inputs[4][bi]
            det_boxes = box_cxcywh_to_xyxy(batched_dets["pred_boxes"][bi])
            det_boxes = det_boxes.cpu().numpy()
            det_scores = (
                batched_dets["pred_scores"][bi].squeeze(1).cpu().numpy().tolist()
            )
            for i in range(org_boxes.shape[0]):
                vis_org.draw_box(org_boxes[i].numpy())
                id_pos = org_boxes[i, :2]
                vis_org.draw_text(
                    str(org_ids[i]), id_pos, horizontal_alignment="left", color="w"
                )
            save_dir = os.path.join(self.cfg.OUTPUT_DIR, "inf_vis")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            vis_org.get_output().save(os.path.join(save_dir, img_id[:-4] + "_org.jpg"))
            for i, score in enumerate(det_scores):
                if score >= thred:
                    b_clr = COLORS[i % len(COLORS)]
                    t_clr = T_COLORS_BG[b_clr]
                    vis_det.draw_box(det_boxes[i], edge_color=b_clr)
                    vis_det.draw_text(
                        "%.2f" % score,
                        det_boxes[i][2:],
                        horizontal_alignment="right",
                        color=t_clr,
                        bg_color=b_clr,
                    )
            vis_det.get_output().save(os.path.join(save_dir, img_id[:-4] + "_det.jpg"))
