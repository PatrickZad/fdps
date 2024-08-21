from psd2.structures.boxes import Boxes
from ...backbone import build_backbone
from ..build import META_ARCH_REGISTRY
import torch
from psd2.structures import ImageList, Instances
from psd2.utils.events import get_event_storage
from psd2.utils.visualizer import Visualizer, mlvl_pca_feat
from psd2.config import configurable
from psd2.modeling.poolers import ROIPooler
from psd2.layers.mem_matching_losses import build_loss_layer
import torch.nn as nn
from psd2.layers.pooling import *
from psd2.modeling.reid_heads.id_assign import build_id_assigner
from psd2.modeling.reid_heads.box_augmentation import build_box_augmentor
import numpy as np
import cv2
import torch.utils.checkpoint as checkpoint
from ...proposal_generator import build_proposal_generator
from ...roi_heads import build_roi_heads


@META_ARCH_REGISTRY.register()
class PsReid_Baseline(nn.Module):
    @configurable
    def __init__(
        self,
        backbone,
        pixel_mean,
        pixel_std,
        input_format,
        vis_period,
        reid_pooler,
        reid_loss,
        pfeat_pooling,
        pid_assigner,
        box_augmentor,
        use_checkpoint,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert (
                input_format is not None
            ), "input_format is required for visualization!"

        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.reid_pooler = reid_pooler
        self.reid_loss = reid_loss
        self.pfeat_pooling = pfeat_pooling
        self.pid_asigner = pid_assigner
        self.box_augmentor = box_augmentor
        self.use_checkpoint = use_checkpoint

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        res = {
            "backbone": backbone,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }
        reid_cfg = cfg.REID_HEAD
        reid_pooler_cfg = reid_cfg.ROI_POOLER
        res["reid_pooler"] = ROIPooler(
            output_size=reid_pooler_cfg.POOLER_RESOLUTION,
            scales=reid_pooler_cfg.SCALES,
            sampling_ratio=reid_pooler_cfg.SAMP_RATIO,
            pooler_type=reid_pooler_cfg.TYPE,
            output_channels=1024,  #  compatible with deformRoiPool
        )
        loss_cfg = reid_cfg.LOSS
        loss_layer = build_loss_layer(loss_cfg, reid_cfg.PERSON_FEATURE.DIM)
        res["reid_loss"] = loss_layer
        pool_layer = reid_cfg.POOL_LAYER
        if pool_layer == "fastavgpool":
            pfeat_pooling = FastGlobalAvgPool2d()
        elif pool_layer == "avgpool":
            pfeat_pooling = nn.AdaptiveAvgPool2d(1)
        elif pool_layer == "maxpool":
            pfeat_pooling = nn.AdaptiveMaxPool2d(1)
        elif pool_layer == "gempoolP":
            pfeat_pooling = GeneralizedMeanPoolingP()  # this one
        elif pool_layer == "gempool":
            pfeat_pooling = GeneralizedMeanPooling()
        elif pool_layer == "avgmaxpool":
            pfeat_pooling = AdaptiveAvgMaxPool2d()
        elif pool_layer == "clipavgpool":
            pfeat_pooling = ClipGlobalAvgPool2d()
        elif pool_layer == "identity":
            pfeat_pooling = nn.Identity()
        elif pool_layer == "flatten":
            pfeat_pooling = Flatten()
        elif pool_layer == "linear":
            pfeat_pooling = FlattenLinear(
                reid_pooler_cfg.POOLER_RESOLUTION[0]
                * reid_pooler_cfg.POOLER_RESOLUTION[1]
                * reid_cfg.PERSON_FEATURE.DIM,
                reid_cfg.PERSON_FEATURE.DIM,
            )
        else:
            raise KeyError(f"{pool_layer} is not supported!")
        res["pfeat_pooling"] = pfeat_pooling
        res["pid_assigner"] = build_id_assigner(reid_cfg.ID_ASSIGN)
        res["box_augmentor"] = (
            build_box_augmentor(reid_cfg.BOX_AUGMENTATION)
            if reid_cfg.BOX_AUGMENTATION.ENABLE
            else None
        )
        res["use_checkpoint"] = reid_cfg.USE_CHECKPOINT
        return res

    @property
    def device(self):
        return self.pixel_mean.device

    def inf_query(self, input_list):
        images, gt_instances = self.preprocess_input(qd["query"] for qd in input_list)
        bk_features = self.backbone(images.tensor)
        reid_bk_feats = self.get_reid_backbone_features(bk_features, images)
        del bk_features
        q_boxes = [gti.gt_boxes.tensor for gti in gt_instances]
        q_featmaps = self.get_reid_person_features(reid_bk_feats, q_boxes)
        del reid_bk_feats
        q_embs = self.get_reid_embed(q_featmaps)
        del q_featmaps
        for bi, feat in enumerate(q_embs):
            input_list[bi]["query"]["feat"] = feat
        return input_list

    def get_reid_embed(self, p_feat_maps):
        raise NotImplementedError

    def get_ps_pos_samples(self, gts):
        pos_boxes, pos_ids, pos_scores = [], [], []
        for gts_i in gts:
            # append gt
            pos_boxes.append(gts_i.gt_boxes.tensor)
            pos_ids.append(gts_i.gt_classes)
            pos_scores.append(
                torch.ones_like(
                    gts_i.gt_classes, dtype=gts_i.gt_boxes.tensor.dtype
                ).unsqueeze(1)
            )  # for vis only
        if self.box_augmentor is not None:
            pos_boxes, pos_ids = self.box_augmentor.augment_boxes(
                pos_boxes,
                pos_ids,
                det_boxes=None,
                det_pids=None,
                img_sizes=[gti.image_size for gti in gts],
            )
            pos_scores = [torch.ones_like(pids).unsqueeze(1) for pids in pos_ids]
        return pos_boxes, pos_ids, pos_scores

    def forward_ps(self, det_backbone_features, image_list, gts):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        # print("reid backbone time: {}".format(time.time() - t0))
        if self.training:
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples(gts)

            pos_featmaps = self.get_reid_person_features(
                reid_bk_feats, pos_boxes
            )  # nf x c x h x w

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    vis_outputs = {
                        "pred_boxes": pos_boxes,
                        "pred_scores": pos_scores,
                        "assign_ids": pos_ids,
                    }
                    self.visualize_training(
                        (image_list, gts),
                        reid_bk_feats,
                        vis_outputs,
                    )
                    self.visualize_training_ps(
                        image_list.tensor,
                        pos_featmaps.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(pos_featmaps)
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            return self.reid_loss(pos_embs, pos_ids, None)
        else:
            num_boxes_per_image = [bi.gt_boxes.tensor.shape[0] for bi in gts]
            p_featmaps = self.get_reid_person_features(
                reid_bk_feats, [bi.gt_boxes.tensor for bi in gts]
            )
            del reid_bk_feats
            p_embs = self.get_reid_embed(p_featmaps)
            del p_featmaps
            p_embs = torch.split(p_embs, num_boxes_per_image)
            # print("reid pred time: {}".format(time.time() - t0))
            return p_embs

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        raise NotImplementedError

    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        raise NotImplementedError

    def forward(self, input_list):
        if self.training:
            images, gt_instances = self.preprocess_input(input_list)
            bk_features = self.backbone(images.tensor)
            for gti in gt_instances:
                cur_pids = gti.gt_classes
                gti.gt_classes[cur_pids == 0] = -1
                gti.gt_classes[cur_pids > 0] -= 2
            return self.forward_ps(bk_features, images, gt_instances)
        else:
            if "query" in input_list[0]:
                return self.inf_query(input_list)
            images, gt_instances = self.preprocess_input(input_list)
            bk_features = self.backbone(images.tensor)
            reid_feats = self.forward_ps(bk_features, images, gt_instances)
            del bk_features
            outputs = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pi, pfeats in enumerate(reid_feats):
                boxes = gt_instances[pi].gt_boxes.tensor
                org_boxes = _resize_boxes(
                    boxes, images.image_sizes[pi], gt_instances[pi]._org_hw
                )
                scores = torch.ones(
                    (boxes.shape[0], 1), dtype=torch.float32, device=boxes.device
                )
                outputs["pred_boxes"].append(org_boxes)
                outputs["pred_scores"].append(scores)
                outputs["reid_feats"].append(pfeats)
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
            org_pids[org_pids > -1] += 2  # labeled people pid +=2, >1
            org_pids[org_pids == -1] = 0  # unlabeled people pid =0
            inst_img.gt_classes = org_pids.long()
            inst_img.gt_classes = org_pids.long()
            inst_img._org_hw = (input_dict["org_height"], input_dict["org_width"])
            inst_img.org_boxes = Boxes(torch.tensor(input_dict["org_boxes"]))
            gt_instances.append(inst_img.to(self.device))
        return (
            ImageList.from_tensors(images, self.backbone.output_shape()[list(self.backbone.output_shape().keys())[-1]].stride),
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
            annos.append(
                {
                    "file_name": inst._file_name,
                    "image_id": inst._image_id,
                    "boxes": inst.gt_boxes.tensor,  # xyxy abs
                    "ids": trans_id,
                }
            )
        bs = len(annos)
        if isinstance(featmap, torch.Tensor):
            featmap = featmap[None]
        if isinstance(featmap, dict):
            featmap = list(featmap.values())
        level_pcas = []
        for featmap_i in featmap:
            level_pcas.append(mlvl_pca_feat(featmap_i[None])[0])
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
                    if "assign_ids" in batched_dets:
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

    @torch.no_grad()
    def visualize_training_ps(self, images, p_feat_maps, p_boxes, p_ids):
        storage = get_event_storage()
        trans_t2img_t = lambda t: t.detach().cpu() * self.pixel_std.cpu().view(
            -1, 1, 1
        ) + self.pixel_mean.cpu().view(-1, 1, 1)
        feat_map_size = p_feat_maps[0].shape[-2:]
        tg_size = [feat_map_size[0] * 16, feat_map_size[1] * 16]
        bs = len(p_boxes)
        for bi in range(bs):
            boxes_bi = p_boxes[bi].cpu()  # n x 4
            box_areas = (boxes_bi[:, 2] - boxes_bi[:, 0]) * boxes_bi[:, 3] - boxes_bi[
                :, 1
            ]
            sort_idxs = torch.argsort(box_areas, dim=0, descending=True)
            feats_bi = p_feat_maps[bi].cpu()  # n x roi_h x roi_w
            idxs = sort_idxs[: min(10, sort_idxs.shape[0])]
            img_rgb_t = trans_t2img_t(images[bi].cpu())  # 3 x h x w
            assigns_on_boxes = []
            for i in idxs:
                assign_on_box = _render_attn_on_box(
                    img_rgb_t * 255, boxes_bi[i], feats_bi[i], tgt_size=tg_size
                )
                assigns_on_boxes.append(assign_on_box)
            cat_assigns_on_boxes = torch.cat(assigns_on_boxes, dim=2)
            storage.put_image("img_{}/attn".format(bi), cat_assigns_on_boxes / 255.0)


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


def _render_attn_on_box(img_rgb_t, pbox_t, feat_box, tgt_size=(384, 192)):
    # NOTE split two view for clearity
    pbox_int = pbox_t.int()  # xyxy
    pbox_img_rgb_t = img_rgb_t[
        :,
        max(pbox_int[1], 0) : min(pbox_int[3] + 1, img_rgb_t.shape[1]),
        max(pbox_int[0], 0) : min(pbox_int[2] + 1, img_rgb_t.shape[2]),
    ].clone()  # 3 x h x w
    pbox_img_rgb_t = torch.nn.functional.interpolate(
        pbox_img_rgb_t[None], size=tgt_size, mode="bilinear", align_corners=False
    ).squeeze(0)

    attn_reshaped = torch.nn.functional.interpolate(
        feat_box[None], size=tgt_size, mode="bilinear", align_corners=False
    ).squeeze(0)
    attn_reshaped = (attn_reshaped**2).sum(0)
    attn_reshaped = (
        255
        * (attn_reshaped - attn_reshaped.min())
        / (attn_reshaped.max() - attn_reshaped.min() + 1e-12)
    )
    attn_img = attn_reshaped.cpu().numpy().astype(np.uint8)
    attn_img = cv2.applyColorMap(attn_img, cv2.COLORMAP_JET)  # bgr hwc
    attn_img = torch.tensor(
        attn_img[..., ::-1].copy(), device=img_rgb_t.device, dtype=img_rgb_t.dtype
    ).permute(2, 0, 1)
    coeff = 0.3
    pbox_img_rgb_t = (1 - coeff) * pbox_img_rgb_t + coeff * attn_img

    return pbox_img_rgb_t


from psd2.modeling.backbone.resnet import (
    ResNet,
    BottleneckBlock,
    DeformBottleneckBlock
)
import torch.nn.init as init
import torch.nn.functional as tF
import copy
from itertools import chain

@META_ARCH_REGISTRY.register()
class PsReid_C4(PsReid_Baseline):
    @configurable
    def __init__(self, bn_neck, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bn_neck = bn_neck
        self.res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "BN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
    @classmethod
    def from_config(cls, cfg):
        res = super().from_config(cfg)
        bn_neck = nn.BatchNorm1d(cfg.REID_HEAD.PERSON_FEATURE.DIM)
        init.constant_(bn_neck.bias, 0.0)
        init.normal_(bn_neck.weight, std=0.01)
        bn_neck.bias.requires_grad_(False)
        res["bn_neck"] = bn_neck

        return res

    def get_reid_backbone_features(self, backbone_features, image_list):
        del image_list
        return backbone_features

    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats = self.reid_pooler([reid_backbone_features["res4"]], d2_boxes)
        roi_feats = (
            checkpoint.checkpoint(self.res5, roi_feats)
            if self.use_checkpoint and self.training
            else self.res5(roi_feats)
        )  # n x c x h x w
        return roi_feats

    def get_reid_embed(self, p_feat_maps):
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        return tF.normalize(pfeat_embs, dim=-1)

@META_ARCH_REGISTRY.register()
class PsReid_C4Next(PsReid_C4):
    @configurable
    def __init__(self, bn_neck, **kwargs) -> None:
        super(PsReid_C4,self).__init__(**kwargs)
        self.bn_neck = bn_neck
    def res5(self,x):
        # for interface compatibility
        x = self.backbone.downsample_layers[3](x)
        x = self.backbone.stages[3](x)
        x = self.backbone.norm3(x)
        return x
    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats = self.reid_pooler([reid_backbone_features["stage3_unorm"]], d2_boxes)
        roi_feats = (
            checkpoint.checkpoint(self.res5, roi_feats)
            if self.use_checkpoint and self.training
            else self.res5(roi_feats)
        )  # n x c x h x w
        return roi_feats
        

@META_ARCH_REGISTRY.register()
class PsReid_C4_2Stream(PsReid_C4): 

    def get_batch_crop_feats(self,crops):
        if isinstance(crops,torch.Tensor):
            bk_features = self.backbone(crops)["res4"]
            bk_features=(
                    checkpoint.checkpoint(self.res5, bk_features)
                    if self.use_checkpoint and self.training
                    else self.res5(bk_features)
                )  # n x c x h x w
            crops_embeds=self.get_reid_embed(bk_features)
        else:
            crops_embeds=[]
            for ci in crops:
                bk_features = self.backbone(ci[None])["res4"]
                bk_features=(
                    checkpoint.checkpoint(self.res5, bk_features)
                    if self.use_checkpoint and self.training
                    else self.res5(bk_features)
                )  # n x c x h x w
                crop_embed=self.get_reid_embed(bk_features)
                crops_embeds.append(crop_embed)
            crops_embeds=torch.cat(crops_embeds,dim=0)
        
        crops_embeds = self.bn_neck(crops_embeds)
        crops_embeds =tF.normalize(crops_embeds, dim=-1)
        return crops_embeds
    def get_reid_embed(self, p_feat_maps):
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        return pfeat_embs

    def get_ms_crop_feats_ids(self,images,gt_instances):
        del images
        pfeats=[]
        n_s1=len(gt_instances[0].crops[0])-1
        s0_crops=[]
        pids0=[]
        for inst in gt_instances:
            s0_crops.extend([crops[0] for crops in inst.crops])
            pids0.append(inst.gt_classes)
        s0_embeds=self.get_batch_crop_feats(s0_crops)
        pids0=torch.cat(pids0)
        pfeats.append(s0_embeds)
        for i in range(1,n_s1+1):
            crops_i=torch.stack(list(chain(*[[crops[i] for crops in inst.crops] for inst in gt_instances])))
            si_embeds=self.get_batch_crop_feats(crops_i)
            pfeats.append(si_embeds)
        pfeats=torch.cat(pfeats,dim=0)
        pids=torch.cat([pids0]*(n_s1+1))
        return pfeats,pids
        
    def preprocess_input(self, input_list):
        if not self.training:
            return super().preprocess_input(input_list)
        images = []
        gt_instances = []
        ulb_offset_start=0
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
            # rearange pids
            num_ulb=(org_pids == -1).sum()
            if num_ulb>0:
                offset_ulb=torch.arange(0,num_ulb,device=self.device)+ulb_offset_start
                ulb_offset_start+=num_ulb
                org_pids[org_pids == -1]-=offset_ulb
            inst_img.gt_classes = org_pids.long()
            inst_img._org_hw = (input_dict["org_height"], input_dict["org_width"])
            inst_img.org_boxes = Boxes(torch.tensor(input_dict["org_boxes"]))
            inst_img.crops=[[crop.to(self.device) for crop in crops] for crops in input_dict["crops"]]
            gt_instances.append(inst_img.to(self.device))
        return (
            ImageList.from_tensors(images, self.backbone.output_shape()[list(self.backbone.output_shape().keys())[-1]].stride),
            gt_instances,
        )

    def forward(self, input_list):
        if self.training:
            images, gt_instances = self.preprocess_input(input_list)
            bk_features = self.backbone(images.tensor)
            crop_embs,crop_ids=self.get_ms_crop_feats_ids(images,gt_instances)
            return self.forward_ps_crops(bk_features, images, gt_instances,crop_embs,crop_ids)
        else:
            return super().forward(input_list)
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        raise NotImplementedError
    def forward_ps_crops(self, det_backbone_features, image_list, gts,crop_embs,crop_ids):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        # print("reid backbone time: {}".format(time.time() - t0))
        if self.training:
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples(gts)

            pos_featmaps = self.get_reid_person_features(
                reid_bk_feats, pos_boxes
            )  # nf x c x h x w

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    vis_outputs = {
                        "pred_boxes": pos_boxes,
                        "pred_scores": pos_scores,
                        "assign_ids": pos_ids,
                    }
                    self.visualize_training(
                        (image_list, gts),
                        reid_bk_feats,
                        vis_outputs,
                    )
                    self.visualize_training_ps(
                        image_list.tensor,
                        pos_featmaps.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(pos_featmaps)
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            losses=self.crop_img_loss(pos_embs,pos_ids,crop_embs,crop_ids)
            pos_ids[pos_ids<0]=-1
            losses.update(self.reid_loss(pos_embs, pos_ids, None))
            return losses
        else:
            num_boxes_per_image = [bi.gt_boxes.tensor.shape[0] for bi in gts]
            p_featmaps = self.get_reid_person_features(
                reid_bk_feats, [bi.gt_boxes.tensor for bi in gts]
            )
            del reid_bk_feats
            p_embs = self.get_reid_embed(p_featmaps)
            del p_featmaps
            p_embs = torch.split(p_embs, num_boxes_per_image)
            # print("reid pred time: {}".format(time.time() - t0))
            return p_embs

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCrops(PsReid_C4):
    @configurable
    def __init__(self, kd_loss_weight, **kwargs) -> None:
        super().__init__(**kwargs)
        self.w_kd=kd_loss_weight
    @classmethod
    def from_config(cls, cfg):
        res = super().from_config(cfg)
        if hasattr(cfg.REID_HEAD.LOSS.LOSS_WEIGHTS,"KD"):
            res["kd_loss_weight"]=cfg.REID_HEAD.LOSS.LOSS_WEIGHTS.KD
        else:
            res["kd_loss_weight"]=0.5
        return res
    def get_crops(self,image_list,boxes):
        tgt_size=(256,128)
        crops=[]
        for bi,boxesi in enumerate(boxes):
            imgi=image_list.tensor[bi]
            h,w=imgi.shape[-2:]
            xmin, ymin, xmax, ymax = boxesi.unbind(1)
            xmin = xmin.clamp(0,w-1).int()
            xmax = xmax.clamp(1,w).ceil().int()
            ymin = ymin.clamp(0,h-1).int()
            ymax = ymax.clamp(1,h).ceil().int()
            cropsi = []
            for x1,y1,x2,y2 in zip(xmin,ymin,xmax,ymax):
                crop_0=imgi[:,y1:y2,x1:x2].clone() # scale org
                crop_0=torch.nn.functional.interpolate(crop_0[None], size=tgt_size, mode='bilinear', align_corners=False)[0]
                cropsi.append(crop_0)
            crops.extend(cropsi)
        return torch.stack(crops)
    def get_crop_embs(self,crops):
        crop_feats=self.backbone(crops)["res4"]
        crop_feats=(
                    checkpoint.checkpoint(self.res5, crop_feats)
                    if self.use_checkpoint and self.training
                    else self.res5(crop_feats)
                )
        crop_feats=self.get_reid_embed(crop_feats)
        return crop_feats
         
    def forward_ps(self, det_backbone_features, image_list, gts):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        # print("reid backbone time: {}".format(time.time() - t0))
        if self.training:
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples(gts)

            pos_featmaps = self.get_reid_person_features(
                reid_bk_feats, pos_boxes
            )  # nf x c x h x w

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    vis_outputs = {
                        "pred_boxes": pos_boxes,
                        "pred_scores": pos_scores,
                        "assign_ids": pos_ids,
                    }
                    self.visualize_training(
                        (image_list, gts),
                        reid_bk_feats,
                        vis_outputs,
                    )
                    self.visualize_training_ps(
                        image_list.tensor,
                        pos_featmaps.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(pos_featmaps)
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            losses=self.reid_loss(pos_embs, pos_ids, None)
            crops=self.get_crops(image_list,pos_boxes)
            crop_embs=self.get_crop_embs(crops)
            crop_ids=pos_ids.clone()
            losses.update(self.crop_img_loss(pos_embs,pos_ids,crop_embs,crop_ids))
            return losses
        else:
            num_boxes_per_image = [bi.gt_boxes.tensor.shape[0] for bi in gts]
            p_featmaps = self.get_reid_person_features(
                reid_bk_feats, [bi.gt_boxes.tensor for bi in gts]
            )
            del reid_bk_feats
            p_embs = self.get_reid_embed(p_featmaps)
            del p_featmaps
            p_embs = torch.split(p_embs, num_boxes_per_image)
            # print("reid pred time: {}".format(time.time() - t0))
            return p_embs

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnSupCon(PsReid_C4_2Stream):
    # TODO deal with BN
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.contrast_loss=SupConLoss()
        self.res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "FrozenBN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        l1=self.contrast_loss(img_embs,crop_embs,img_pids,crop_pids)
        l2=self.contrast_loss(crop_embs,img_embs,crop_pids,img_pids)
        return {"loss_img2crop":l1*0.5,"loss_crop2img":l2*0.5}
@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamUni(PsReid_C4_2Stream):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.contrast_loss=SupConLoss()
        self.res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "FrozenBN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )

    def forward_ps_crops(self, det_backbone_features, image_list, gts,crop_embs,crop_ids):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        # print("reid backbone time: {}".format(time.time() - t0))
        if self.training:
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples(gts)

            pos_featmaps = self.get_reid_person_features(
                reid_bk_feats, pos_boxes
            )  # nf x c x h x w

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    vis_outputs = {
                        "pred_boxes": pos_boxes,
                        "pred_scores": pos_scores,
                        "assign_ids": pos_ids,
                    }
                    self.visualize_training(
                        (image_list, gts),
                        reid_bk_feats,
                        vis_outputs,
                    )
                    self.visualize_training_ps(
                        image_list.tensor,
                        pos_featmaps.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(pos_featmaps)
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            pos_embs=torch.cat([pos_embs,crop_embs],dim=0)
            pos_ids=torch.cat([pos_ids,crop_ids])
            pos_ids[pos_ids<0]=-1
            losses=self.reid_loss(pos_embs, pos_ids, None)
            return losses
        else:
            num_boxes_per_image = [bi.gt_boxes.tensor.shape[0] for bi in gts]
            p_featmaps = self.get_reid_person_features(
                reid_bk_feats, [bi.gt_boxes.tensor for bi in gts]
            )
            del reid_bk_feats
            p_embs = self.get_reid_embed(p_featmaps)
            del p_featmaps
            p_embs = torch.split(p_embs, num_boxes_per_image)
            # print("reid pred time: {}".format(time.time() - t0))
            return p_embs

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamProtoCon(PsReid_C4_2Stream):
    # TODO deal with BN
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.contrast_loss=ProtoConLoss()
        self.res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "FrozenBN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
    def get_ms_crop_feats_ids(self,images,gt_instances):
        del images
        pfeats=[]
        n_s1=len(gt_instances[0].crops[0])-1
        s0_crops=[]
        all_pids=torch.cat([inst.gt_classes for inst in gt_instances])
        if (all_pids>-1).sum()==0:
            return None,None
        else:
            pos_mask=all_pids>-1
            for inst in gt_instances:
                cur_crops=[crops[0] for crops in inst.crops]
                for pid,pcrop in zip(inst.gt_classes.list(),cur_crops):
                    if pid>-1:
                        s0_crops.append(pcrop)
                s0_embeds=self.get_batch_crop_feats(s0_crops)
                pfeats.append(s0_embeds)
                for i in range(1,n_s1+1):
                    crops_i=torch.stack(list(chain(*[[crops[i] for crops in inst.crops] for inst in gt_instances])))
                    crops_i=crops_i[pos_mask]
                    si_embeds=self.get_batch_crop_feats(crops_i)
                    pfeats.append(si_embeds)
                pfeats=torch.cat(pfeats,dim=0)
                pids=torch.cat([all_pids[pos_mask]]*(n_s1+1))
            return pfeats,pids
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        del img_embs
        proto=torch.cat([self.reid_loss.lookup_table,self.reid_loss.queue],dim=0)
        l_con=self.contrast_loss(crop_embs,proto,crop_pids)
        return {"loss_contrast":l_con}

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamFixOrg(PsReid_C4_2Stream):
    # TODO deal with BN
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.crop_loss=copy.deepcopy(self.reid_loss)
    def get_ms_crop_feats_ids(self,images,gt_instances):
        del images
        s0_crops=[]
        all_pids=torch.cat([inst.gt_classes for inst in gt_instances])
        for inst in gt_instances:
            cur_crops=[crops[0] for crops in inst.crops]
            s0_crops.extend(cur_crops)
        s0_crops=torch.stack(s0_crops)
        s0_embeds=self.get_batch_crop_feats(s0_crops)
        return s0_embeds,all_pids
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        del img_embs
        pids=crop_pids.clone()
        pids[pids<0]=-1
        loss=self.crop_loss(crop_embs,pids,None)
        return {k+"_crops": v for k,v in loss.items()}

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOim(PsReid_C4_2StreamOnCrops):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.crop_loss=copy.deepcopy(self.reid_loss)
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        del img_embs
        del img_pids
        loss=self.crop_loss(crop_embs,crop_pids,None)
        return {k+"_crops": v for k,v in loss.items()}

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOimProtoCon(PsReid_C4_2StreamOnCrops):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.crop_loss=copy.deepcopy(self.reid_loss)
        self.contrast_loss=ProtoConLoss()
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        loss=self.crop_loss(crop_embs,crop_pids,None)
        losses={k+"_crops": v for k,v in loss.items()}
        pos_mask=crop_pids>-1
        if pos_mask.sum().item()==0:
            losses["loss_contrast_crop"]=crop_embs.mean()* 0.0
            losses["loss_contrast_image"]=img_embs.mean()*0.0
        else:
            crop_embs=crop_embs[pos_mask]
            crop_pids=crop_pids[pos_mask]
            proto_img=torch.cat([self.reid_loss.lb_layer.lookup_table,self.reid_loss.ulb_layer.queue],dim=0)
            l_con=self.contrast_loss(crop_embs,proto_img,crop_pids)*self.w_kd
            losses["loss_contrast_crop"]=l_con
            proto_crop=torch.cat([self.crop_loss.lb_layer.lookup_table,self.crop_loss.ulb_layer.queue],dim=0)
            pos_mask=img_pids>-1
            img_embs=img_embs[pos_mask]
            img_pids=img_pids[pos_mask]
            l_con=self.contrast_loss(img_embs,proto_crop,img_pids)*self.w_kd
            losses["loss_contrast_image"]=l_con
        return losses

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOimLbProtoCon(PsReid_C4_2StreamOnCrops):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.crop_loss=copy.deepcopy(self.reid_loss)
        self.contrast_loss=ProtoConLoss()
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        loss=self.crop_loss(crop_embs,crop_pids,None)
        losses={k+"_crops": v for k,v in loss.items()}
        pos_mask=crop_pids>-1
        if pos_mask.sum().item()==0:
            losses["loss_contrast_crop"]=crop_embs.mean()* 0.0
            losses["loss_contrast_image"]=img_embs.mean()*0.0
        else:
            crop_embs=crop_embs[pos_mask]
            crop_pids=crop_pids[pos_mask]
            proto_img=self.reid_loss.lb_layer.lookup_table
            l_con=self.contrast_loss(crop_embs,proto_img,crop_pids)*self.w_kd
            losses["loss_contrast_crop"]=l_con
            proto_crop=self.crop_loss.lb_layer.lookup_table
            pos_mask=img_pids>-1
            img_embs=img_embs[pos_mask]
            img_pids=img_pids[pos_mask]
            l_con=self.contrast_loss(img_embs,proto_crop,img_pids)*self.w_kd
            losses["loss_contrast_image"]=l_con
        return losses

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOimProtoTrip(PsReid_C4_2StreamOnCrops):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.crop_loss=copy.deepcopy(self.reid_loss)
        self.triplet_loss = TripletLoss(0.3, "mean")
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        loss=self.crop_loss(crop_embs,crop_pids,None)
        losses={k+"_crops": v for k,v in loss.items()}
        pos_mask=crop_pids>-1
        if pos_mask.sum().item()==0:
            losses["loss_trip_crop"]=crop_embs.mean()* 0.0
            losses["loss_trip_image"]=img_embs.mean()*0.0
        else:
            crop_embs=crop_embs[pos_mask]
            crop_pids=crop_pids[pos_mask]
            proto_img=torch.cat([self.reid_loss.lb_layer.lookup_table,self.reid_loss.ulb_layer.queue],dim=0)
            proto_ids = torch.arange(
                0, proto_img.shape[0], dtype=crop_pids.dtype, device=crop_pids.device
            )
            trip = self.triplet_loss(
                    crop_embs,
                    proto_img,
                    crop_pids,
                    proto_ids,
                    normalize_feature=True,
                )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k+"_crop"] = v*self.w_kd
            proto_crop=torch.cat([self.crop_loss.lb_layer.lookup_table,self.crop_loss.ulb_layer.queue],dim=0)
            pos_mask=img_pids>-1
            img_embs=img_embs[pos_mask]
            img_pids=img_pids[pos_mask]
            proto_ids = torch.arange(
                0, proto_crop.shape[0], dtype=proto_ids.dtype, device=proto_ids.device
            )
            trip = self.triplet_loss(
                    img_embs,
                    proto_crop,
                    img_pids,
                    proto_ids,
                    normalize_feature=True,
                )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k+"_image"] = v*self.w_kd
        return losses

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOimLbProtoTrip(PsReid_C4_2StreamOnCrops):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.crop_loss=copy.deepcopy(self.reid_loss)
        self.triplet_loss = TripletLoss(0.3, "mean")
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        loss=self.crop_loss(crop_embs,crop_pids,None)
        losses={k+"_crops": v for k,v in loss.items()}
        pos_mask=crop_pids>-1
        if pos_mask.sum().item()==0:
            losses["loss_trip_crop"]=crop_embs.mean()* 0.0
            losses["loss_trip_image"]=img_embs.mean()*0.0
        else:
            crop_embs=crop_embs[pos_mask]
            crop_pids=crop_pids[pos_mask]
            proto_img=self.reid_loss.lb_layer.lookup_table
            proto_ids = torch.arange(
                0, proto_img.shape[0], dtype=crop_pids.dtype, device=crop_pids.device
            )
            trip = self.triplet_loss(
                    crop_embs,
                    proto_img,
                    crop_pids,
                    proto_ids,
                    normalize_feature=True,
                )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k+"_crop"] = v*self.w_kd
            proto_crop=self.crop_loss.lb_layer.lookup_table
            pos_mask=img_pids>-1
            img_embs=img_embs[pos_mask]
            img_pids=img_pids[pos_mask]
            proto_ids = torch.arange(
                0, proto_crop.shape[0], dtype=proto_ids.dtype, device=proto_ids.device
            )
            trip = self.triplet_loss(
                    img_embs,
                    proto_crop,
                    img_pids,
                    proto_ids,
                    normalize_feature=True,
                )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k+"_image"] = v*self.w_kd
        return losses

import psd2.utils.comm as comm
@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOimProtoConDist(PsReid_C4_2StreamOnCropsOimProtoCon):
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        loss=self.crop_loss(crop_embs,crop_pids,None)
        losses={k+"_crops": v for k,v in loss.items()}
        pos_mask=crop_pids>-1
        cur_num_lb=pos_mask.sum()
        comm.synchronize()
        all_num_lb = comm.all_gather(cur_num_lb)
        num_lb = sum([num.to("cpu") for num in all_num_lb])
        num_lb = torch.clamp(num_lb / comm.get_world_size(), min=1).item()
        if cur_num_lb.item()==0:
            losses["loss_contrast_crop"]=crop_embs.mean()* 0.0
            losses["loss_contrast_image"]=img_embs.mean()*0.0
        else:
            crop_embs=crop_embs[pos_mask]
            crop_pids=crop_pids[pos_mask]
            proto_img=torch.cat([self.reid_loss.lb_layer.lookup_table,self.reid_loss.ulb_layer.queue],dim=0)
            l_con=self.contrast_loss(crop_embs,proto_img,crop_pids)*self.w_kd
            losses["loss_contrast_crop"]=l_con*cur_num_lb/num_lb
            proto_crop=torch.cat([self.crop_loss.lb_layer.lookup_table,self.crop_loss.ulb_layer.queue],dim=0)
            pos_mask=img_pids>-1
            img_embs=img_embs[pos_mask]
            img_pids=img_pids[pos_mask]
            l_con=self.contrast_loss(img_embs,proto_crop,img_pids)*self.w_kd
            losses["loss_contrast_image"]=l_con*cur_num_lb/num_lb
        return losses

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOimProtoConRe(PsReid_C4_2StreamOnCropsOimProtoCon):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.re=RandomErasingAfterNorm(self.pixel_mean[:,0,0],self.pixel_std[:,0,0])
    
    def get_crops(self,image_list,boxes):
        tgt_size=(256,128)
        crops=[]
        for bi,boxesi in enumerate(boxes):
            imgi=image_list.tensor[bi]
            h,w=imgi.shape[-2:]
            xmin, ymin, xmax, ymax = boxesi.unbind(1)
            xmin = xmin.clamp(0,w-1).int()
            xmax = xmax.clamp(1,w).ceil().int()
            ymin = ymin.clamp(0,h-1).int()
            ymax = ymax.clamp(1,h).ceil().int()
            cropsi = []
            for x1,y1,x2,y2 in zip(xmin,ymin,xmax,ymax):
                crop_0=imgi[:,y1:y2,x1:x2].clone() # scale org
                crop_0=torch.nn.functional.interpolate(crop_0[None], size=tgt_size, mode='bilinear', align_corners=False)[0]
                crop_0=self.re(crop_0)
                cropsi.append(crop_0)
            crops.extend(cropsi)
        return torch.stack(crops)
import torchvision.transforms.functional as tvF
@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOimProtoConReFlip(PsReid_C4_2StreamOnCropsOimProtoConRe):
    def get_crops(self,image_list,boxes):
        tgt_size=(256,128)
        crops=[]
        for bi,boxesi in enumerate(boxes):
            imgi=image_list.tensor[bi]
            h,w=imgi.shape[-2:]
            xmin, ymin, xmax, ymax = boxesi.unbind(1)
            xmin = xmin.clamp(0,w-1).int()
            xmax = xmax.clamp(1,w).ceil().int()
            ymin = ymin.clamp(0,h-1).int()
            ymax = ymax.clamp(1,h).ceil().int()
            cropsi = []
            for x1,y1,x2,y2 in zip(xmin,ymin,xmax,ymax):
                crop_0=imgi[:,y1:y2,x1:x2].clone() # scale org
                crop_0=torch.nn.functional.interpolate(crop_0[None], size=tgt_size, mode='bilinear', align_corners=False)[0]
                if torch.rand(1) < 0.5:
                    crop_0=tvF.hflip(crop_0)
                crop_0=self.re(crop_0)
                cropsi.append(crop_0)
            crops.extend(cropsi)
        return torch.stack(crops)

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOimLbProtoConRe(PsReid_C4_2StreamOnCropsOimLbProtoCon):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.re=RandomErasingAfterNorm(self.pixel_mean[:,0,0],self.pixel_std[:,0,0])
    
    def get_crops(self,image_list,boxes):
        tgt_size=(256,128)
        crops=[]
        for bi,boxesi in enumerate(boxes):
            imgi=image_list.tensor[bi]
            h,w=imgi.shape[-2:]
            xmin, ymin, xmax, ymax = boxesi.unbind(1)
            xmin = xmin.clamp(0,w-1).int()
            xmax = xmax.clamp(1,w).ceil().int()
            ymin = ymin.clamp(0,h-1).int()
            ymax = ymax.clamp(1,h).ceil().int()
            cropsi = []
            for x1,y1,x2,y2 in zip(xmin,ymin,xmax,ymax):
                crop_0=imgi[:,y1:y2,x1:x2].clone() # scale org
                crop_0=torch.nn.functional.interpolate(crop_0[None], size=tgt_size, mode='bilinear', align_corners=False)[0]
                crop_0=self.re(crop_0)
                cropsi.append(crop_0)
            crops.extend(cropsi)
        return torch.stack(crops)
@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOimOnlineCon(PsReid_C4_2StreamOnCrops):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.crop_loss=copy.deepcopy(self.reid_loss)
        self.contrast_loss=SupConLoss()
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        loss=self.crop_loss(crop_embs,crop_pids,None)
        losses={k+"_crops": v for k,v in loss.items()}
        pos_mask=crop_pids>-1
        if pos_mask.sum().item()==0:
            losses["loss_contrast_crop"]=crop_embs.mean()* 0.0
            losses["loss_contrast_image"]=img_embs.mean()*0.0
        else:
            crop_embs_pos=crop_embs[pos_mask]
            crop_pids_pos=crop_pids[pos_mask]
            l_con=self.contrast_loss(crop_embs_pos,img_embs,crop_pids_pos,img_pids)*0.5
            losses["loss_contrast_crop"]=l_con
            pos_mask=img_pids>-1
            img_embs_pos=img_embs[pos_mask]
            img_pids_pos=img_pids[pos_mask]
            l_con=self.contrast_loss(img_embs_pos,crop_embs,img_pids_pos,crop_pids)*0.5
            losses["loss_contrast_image"]=l_con
        return losses    

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOimOnlineTrip(PsReid_C4_2StreamOnCrops):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.crop_loss=copy.deepcopy(self.reid_loss)
        self.triplet_loss = TripletLoss(0.3, "mean")
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        loss=self.crop_loss(crop_embs,crop_pids,None)
        losses={k+"_crops": v for k,v in loss.items()}
        pos_mask=crop_pids>-1
        if pos_mask.sum().item()==0:
            losses["loss_trip_crop"]=crop_embs.mean()* 0.0
            losses["loss_trip_image"]=img_embs.mean()*0.0
        else:
            crop_embs_pos=crop_embs[pos_mask]
            crop_pids_pos=crop_pids[pos_mask]
            trip = self.triplet_loss(
                    crop_embs_pos,
                    img_embs,
                    crop_pids_pos,
                    img_pids,
                    normalize_feature=True,
                )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k+"_crop"] = v*self.w_kd
            pos_mask=img_pids>-1
            img_embs_pos=img_embs[pos_mask]
            img_pids_pos=img_pids[pos_mask]
            trip = self.triplet_loss(
                    img_embs_pos,
                    crop_embs,
                    img_pids_pos,
                    crop_pids,
                    normalize_feature=True,
                )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k+"_image"] = v*self.w_kd
        return losses  

@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOimOnlineConRe(PsReid_C4_2StreamOnCropsOimOnlineCon):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.re=RandomErasingAfterNorm(self.pixel_mean[:,0,0],self.pixel_std[:,0,0])
    def get_crops(self,image_list,boxes):
        tgt_size=(256,128)
        crops=[]
        for bi,boxesi in enumerate(boxes):
            imgi=image_list.tensor[bi]
            h,w=imgi.shape[-2:]
            xmin, ymin, xmax, ymax = boxesi.unbind(1)
            xmin = xmin.clamp(0,w-1).int()
            xmax = xmax.clamp(1,w).ceil().int()
            ymin = ymin.clamp(0,h-1).int()
            ymax = ymax.clamp(1,h).ceil().int()
            cropsi = []
            for x1,y1,x2,y2 in zip(xmin,ymin,xmax,ymax):
                crop_0=imgi[:,y1:y2,x1:x2].clone() # scale org
                crop_0=torch.nn.functional.interpolate(crop_0[None], size=tgt_size, mode='bilinear', align_corners=False)[0]
                crop_0=self.re(crop_0)
                cropsi.append(crop_0)
            crops.extend(cropsi)
        return torch.stack(crops)
@META_ARCH_REGISTRY.register()
class PsReid_C4_2StreamOnCropsOimCombineCon(PsReid_C4_2StreamOnCrops):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.crop_loss=copy.deepcopy(self.reid_loss)
        self.contrast_loss_online=SupConLoss()
        self.contrast_loss=ProtoConLoss()
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        loss=self.crop_loss(crop_embs,crop_pids,None)
        losses={k+"_crops": v for k,v in loss.items()}
        pos_mask=crop_pids>-1
        if pos_mask.sum().item()==0:
            losses["loss_contrast_crop"]=crop_embs.mean()* 0.0
            losses["loss_contrast_image"]=img_embs.mean()*0.0
            losses["loss_contrast_crop_online"]=crop_embs.mean()* 0.0
            losses["loss_contrast_image_online"]=img_embs.mean()*0.0
        else:
            crop_embs_pos=crop_embs[pos_mask]
            crop_pids_pos=crop_pids[pos_mask]
            l_con=self.contrast_loss_online(crop_embs_pos,img_embs,crop_pids_pos,img_pids)*0.25
            losses["loss_contrast_crop_online"]=l_con
            proto_img=torch.cat([self.reid_loss.lb_layer.lookup_table,self.reid_loss.ulb_layer.queue],dim=0)
            l_con=self.contrast_loss(crop_embs_pos,proto_img,crop_pids_pos)*0.25
            losses["loss_contrast_crop"]=l_con
            pos_mask=img_pids>-1
            img_embs_pos=img_embs[pos_mask]
            img_pids_pos=img_pids[pos_mask]
            l_con=self.contrast_loss_online(img_embs_pos,crop_embs,img_pids_pos,crop_pids)*0.25
            losses["loss_contrast_image_online"]=l_con
            proto_crop=torch.cat([self.crop_loss.lb_layer.lookup_table,self.crop_loss.ulb_layer.queue],dim=0)
            l_con=self.contrast_loss(img_embs_pos,proto_crop,img_pids_pos)*0.25
            losses["loss_contrast_image"]=l_con
        return losses    


@META_ARCH_REGISTRY.register()
class PsReid_C4_D(PsReid_C4):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "BN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "deform_modulated": True,
                    "deform_num_groups": 1,
                    "block_class": DeformBottleneckBlock,
                }
            )
        )

from psd2.layers.metric_loss import TripletLoss


@META_ARCH_REGISTRY.register()
class PsReid_C4Trip1(PsReid_C4):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.triplet_loss = TripletLoss(0.3, "mean")

    def forward_ps(self, det_backbone_features, image_list, gts):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        # print("reid backbone time: {}".format(time.time() - t0))
        if self.training:
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples(gts)

            pos_featmaps = self.get_reid_person_features(
                reid_bk_feats, pos_boxes
            )  # nf x c x h x w

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    vis_outputs = {
                        "pred_boxes": pos_boxes,
                        "pred_scores": pos_scores,
                        "assign_ids": pos_ids,
                    }
                    self.visualize_training(
                        (image_list, gts),
                        reid_bk_feats,
                        vis_outputs,
                    )
                    self.visualize_training_ps(
                        image_list.tensor,
                        pos_featmaps.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(
                pos_featmaps
            )  # before-bnneck is inccorect,for the lookup-table are after-bnneck
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            reid_loss = self.reid_loss(pos_embs, pos_ids, None)
            oim_lookup = self.reid_loss.lb_layer.lookup_table
            lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=pos_ids.dtype, device=pos_ids.device
            )
            feats1 = pos_embs[pos_ids > -1]
            feats2 = torch.cat([pos_embs[pos_ids > -1], oim_lookup], dim=0)
            if feats1.shape[0] < 1:
                reid_loss["loss_triplet"] = torch.zeros(1, device=self.device)
            else:
                trip = self.triplet_loss(
                    feats1,
                    feats2,
                    pos_ids[pos_ids > -1],
                    torch.cat([pos_ids[pos_ids > -1], lookup_ids], dim=0),
                    normalize_feature=True,
                )
                for k, v in trip.items():
                    if "loss" in k:
                        reid_loss[k] = v
                    else:
                        get_event_storage().put_scalar(k, v)
            return reid_loss
        else:
            num_boxes_per_image = [bi.gt_boxes.tensor.shape[0] for bi in gts]
            p_featmaps = self.get_reid_person_features(
                reid_bk_feats, [bi.gt_boxes.tensor for bi in gts]
            )
            del reid_bk_feats
            p_embs = self.get_reid_embed(p_featmaps)
            del p_featmaps
            p_embs = torch.split(p_embs, num_boxes_per_image)
            # print("reid pred time: {}".format(time.time() - t0))
            return p_embs

@META_ARCH_REGISTRY.register()
class PsReid_C4Trip1_D(PsReid_C4Trip1):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "BN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "deform_modulated": True,
                    "deform_num_groups": 1,
                    "block_class": DeformBottleneckBlock,
                }
            )
        )


@META_ARCH_REGISTRY.register()
class PsReid_C4Trip3(PsReid_C4):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.triplet_loss = TripletLoss(None, "mean")

    def forward_ps(self, det_backbone_features, image_list, gts):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        # print("reid backbone time: {}".format(time.time() - t0))
        if self.training:
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples(gts)

            pos_featmaps = self.get_reid_person_features(
                reid_bk_feats, pos_boxes
            )  # nf x c x h x w

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    vis_outputs = {
                        "pred_boxes": pos_boxes,
                        "pred_scores": pos_scores,
                        "assign_ids": pos_ids,
                    }
                    self.visualize_training(
                        (image_list, gts),
                        reid_bk_feats,
                        vis_outputs,
                    )
                    self.visualize_training_ps(
                        image_list.tensor,
                        pos_featmaps.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(
                pos_featmaps
            )  # before-bnneck is inccorect,for the lookup-table are after-bnneck
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            reid_loss = self.reid_loss(pos_embs, pos_ids, None)
            oim_lookup = self.reid_loss.lb_layer.lookup_table
            lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=pos_ids.dtype, device=pos_ids.device
            )
            feats1 = pos_embs[pos_ids > -1]
            feats2 = torch.cat([pos_embs[pos_ids > -1], oim_lookup], dim=0)
            if feats1.shape[0] < 1:
                reid_loss["loss_triplet"] = torch.zeros(1, device=self.device)
            else:
                trip = self.triplet_loss(
                    feats1,
                    feats2,
                    pos_ids[pos_ids > -1],
                    torch.cat([pos_ids[pos_ids > -1], lookup_ids], dim=0),
                    normalize_feature=True,
                    hard_mining=False,
                )
                for k, v in trip.items():
                    if "loss" in k:
                        reid_loss[k] = v
                    else:
                        get_event_storage().put_scalar(k, v)
            return reid_loss
        else:
            num_boxes_per_image = [bi.gt_boxes.tensor.shape[0] for bi in gts]
            p_featmaps = self.get_reid_person_features(
                reid_bk_feats, [bi.gt_boxes.tensor for bi in gts]
            )
            del reid_bk_feats
            p_embs = self.get_reid_embed(p_featmaps)
            del p_featmaps
            p_embs = torch.split(p_embs, num_boxes_per_image)
            # print("reid pred time: {}".format(time.time() - t0))
            return p_embs


@META_ARCH_REGISTRY.register()
class PsReid_C4_SideRCNN(PsReid_C4):
    @configurable
    def __init__(
        self,
        proposal_generator,
        roi_heads,
        side_init,
        cws,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.side_init = side_init
        self.alpha_res3 = nn.Parameter(torch.tensor(0.0))
        self.alpha_res4 = nn.Parameter(torch.tensor(0.0))
        # side reid network
        self.side_res3 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 4,
                    "stride_per_block": [2, 1, 1, 1],
                    "in_channels": 256,
                    "out_channels": 512,
                    "norm": "FrozenBN",
                    "bottleneck_channels": 128,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        self.side_res4 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 6,
                    "stride_per_block": [2, 1, 1, 1, 1, 1],
                    "in_channels": 512,
                    "out_channels": 1024,
                    "norm": "FrozenBN",
                    "bottleneck_channels": 256,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        self.cws = cws
        # freeze reid params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.pfeat_pooling.parameters():
            p.requires_grad_(False)
        for p in self.bn_neck.parameters():
            p.requires_grad_(False)

    def load_state_dict(self, *args, **kws):
        output = super().load_state_dict(*args, **kws)
        model_resume = True
        for param_k in output.missing_keys:
            if "side_res" in param_k:
                model_resume = False
                break
        if not model_resume and self.side_init != "":
            # model is not resuming from saved checkpoint
            # init side networks
            state_dict = _load_file(self.side_init)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            side_res_params = [{}, {}, {}]
            for si in range(3):
                res_name = "res{}".format(si + 3)
                for pn, pa in state_dict.items():
                    if res_name in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        side_res_params[si][pn.split(res_name)[1][1:]] = pa
            self.side_res3.load_state_dict(side_res_params[0],strict=False)
            self.side_res4.load_state_dict(side_res_params[1],strict=False)
            self.roi_heads.res5.load_state_dict(side_res_params[2],strict=False)
        return output

    def train(self, mode=True):
        self.training = mode
        if mode:
            for n, m in self.named_children():
                if "backbone" in n or "reid_pooler" in n or "bn_neck" in n:
                    m.eval()
                else:
                    m.train()
        else:
            for m in self.children():
                m.eval()

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        backbone = ret["backbone"]
        ret.update(
            {
                "proposal_generator": build_proposal_generator(
                    cfg, backbone.output_shape()
                ),
                "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
                "side_init": cfg.REID_HEAD.INIT_WEIGHT,
                "cws": cfg.CWS,
            }
        )
        return ret

    def det_backbone(self, reid_backbone_features):
        det_res3_feat = (
            checkpoint.checkpoint(self.side_res3, reid_backbone_features["res2"])
            if self.use_checkpoint and self.training
            else self.side_res3(reid_backbone_features["res2"])
        )
        alpha = torch.sigmoid(self.alpha_res3)
        det_res3_feat = (
            alpha * reid_backbone_features["res3"] + (1 - alpha) * det_res3_feat
        )
        det_res4_feat = (
            checkpoint.checkpoint(self.side_res4, det_res3_feat)
            if self.use_checkpoint and self.training
            else self.side_res4(det_res3_feat)
        )
        del det_res3_feat
        alpha = torch.sigmoid(self.alpha_res4)
        det_res4_feat = (
            alpha * reid_backbone_features["res4"] + (1 - alpha) * det_res4_feat
        )
        return {"res4": det_res4_feat}

    def forward_det(self, image_list, reid_bk_features, gt_instances):
        """
        return det feat maps and predictions
        """
        features = self.det_backbone(reid_bk_features)
        if self.training:
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            det_pred_instances, det_losses = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    outputs = {"pred_boxes": [], "pred_scores": []}
                    for pi, pred_img in enumerate(det_pred_instances):
                        boxes = pred_img.pred_boxes.tensor
                        scores = pred_img.scores
                        outputs["pred_boxes"].append(boxes)
                        outputs["pred_scores"].append(scores.unsqueeze(1))
                    for gti in gt_instances:
                        gt_id = gti.gt_classes
                        gt_id[gt_id == 0] = -1
                        gt_id[gt_id > 0] -= 2
                    self.visualize_training(
                        (image_list, gt_instances),
                        features[list(features.keys())[-1]],
                        outputs,
                    )
            losses = {}
            losses.update(det_losses)
            losses.update(proposal_losses)
            return losses
        else:
            proposals, _ = self.proposal_generator(image_list, features, None)
            det_pred_instances, _ = self.roi_heads(
                image_list, features, proposals, None
            )
            # print("det pred time: {}".format(time.time() - t0))
            det_outputs = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pi, pred_img in enumerate(det_pred_instances):
                boxes = pred_img.pred_boxes.tensor
                scores = pred_img.scores
                det_outputs["pred_boxes"].append(boxes)
                det_outputs["pred_scores"].append(scores.unsqueeze(1))
            return det_outputs

    def forward(self, input_list):
        if self.training:
            images, gt_instances = self.preprocess_input(input_list)
            bk_features = self.backbone(images.tensor)
            return self.forward_det(images, bk_features, gt_instances)
        else:
            if "query" in input_list[0]:
                return self.inf_query(input_list)
            images, gt_instances = self.preprocess_input(input_list)
            bk_features = self.backbone(images.tensor)
            det_pred = self.forward_det(images, bk_features, gt_instances)
            reid_feats = self.forward_ps(det_pred, bk_features, images, gt_instances)
            del bk_features
            outputs = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pi, pfeats in enumerate(reid_feats):
                boxes = det_pred["pred_boxes"][pi]
                org_boxes = _resize_boxes(
                    boxes, images.image_sizes[pi], gt_instances[pi]._org_hw
                )
                scores = det_pred["pred_scores"][pi]
                outputs["pred_boxes"].append(org_boxes)
                outputs["pred_scores"].append(scores)
                outputs["reid_feats"].append(pfeats)
            return outputs

    def forward_ps(self, det_pred, reid_bk_feats, image_list, gts):
        num_boxes_per_image = [bi.shape[0] for bi in det_pred["pred_boxes"]]
        p_featmaps = self.get_reid_person_features(
            reid_bk_feats, det_pred["pred_boxes"]
        )
        del reid_bk_feats
        p_embs = self.get_reid_embed(p_featmaps)
        del p_featmaps
        if self.cws:
            cat_scores = torch.cat(det_pred["pred_scores"], dim=0)
            p_embs *= cat_scores  # .unsqueeze(1)
        p_embs = torch.split(p_embs, num_boxes_per_image)
        # print("reid pred time: {}".format(time.time() - t0))
        return p_embs

@META_ARCH_REGISTRY.register()
class PsReid_C4_D_SideLiteRCNN(PsReid_C4_SideRCNN):
    @configurable
    def __init__(
        self,
        proposal_generator,
        roi_heads,
        side_init,
        cws,
        *args,
        **kws,
    ):
        super(PsReid_C4_SideRCNN,self).__init__(*args, **kws)
        self.res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "BN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "deform_modulated": True,
                    "deform_num_groups": 1,
                    "block_class": DeformBottleneckBlock,
                }
            )
        )
        lite_res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 2,
                    "stride_per_block": [1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "BN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        roi_heads.res5=lite_res5
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.side_init = side_init
        self.alpha_res3 = nn.Parameter(torch.tensor(0.0))
        self.alpha_res4 = nn.Parameter(torch.tensor(0.0))
        # side reid network
        self.side_res3 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 2,
                    "stride_per_block": [2, 1],
                    "in_channels": 256,
                    "out_channels": 512,
                    "norm": "BN",
                    "bottleneck_channels": 128,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        self.side_res4 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 2,
                    "stride_per_block": [2, 1],
                    "in_channels": 512,
                    "out_channels": 1024,
                    "norm": "BN",
                    "bottleneck_channels": 256,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        self.cws = cws
        # freeze reid params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.pfeat_pooling.parameters():
            p.requires_grad_(False)
        for p in self.bn_neck.parameters():
            p.requires_grad_(False)

from psd2.layers import Conv2d, ShapeSpec
from psd2.modeling.backbone.resnet import BasicBlock


@META_ARCH_REGISTRY.register()
class PsReid_C4_SideRCNN_R34(PsReid_C4_SideRCNN):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        bk_out_shape = self.backbone.output_shape()
        self.lateral_conv2 = Conv2d(
            bk_out_shape["res2"].channels,
            64,
            kernel_size=1,
            bias=False,
        )  # same with fpn
        self.lateral_conv3 = Conv2d(
            bk_out_shape["res3"].channels,
            128,
            kernel_size=1,
            bias=False,
        )  # same with fpn
        self.lateral_conv4 = Conv2d(
            bk_out_shape["res4"].channels,
            256,
            kernel_size=1,
            bias=False,
        )  # same with fpn
        self.side_res3 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 4,
                    "stride_per_block": [2, 1, 1, 1],
                    "in_channels": 64,
                    "out_channels": 128,
                    "norm": "BN",
                    "block_class": BasicBlock,
                }
            )
        )
        self.side_res4 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 6,
                    "stride_per_block": [2, 1, 1, 1, 1, 1],
                    "in_channels": 128,
                    "out_channels": 256,
                    "norm": "BN",
                    "block_class": BasicBlock,
                }
            )
        )

    @classmethod
    def det_out_shape(self):
        out_feature_channels = {"res3": 128, "res4": 256, "res5": 512}
        out_feature_strides = {"res3": 8, "res4": 16, "res5": 32}
        return {
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res3", "res4", "res5"]
        }

    @classmethod
    def from_config(cls, cfg):
        ret = super(PsReid_C4_SideRCNN, cls).from_config(cfg)
        cfg.defrost()
        cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
        cfg.MODEL.RESNETS.DEPTH = 34
        cfg.freeze()
        ret.update(
            {
                "proposal_generator": build_proposal_generator(
                    cfg, cls.det_out_shape()
                ),
                "roi_heads": build_roi_heads(cfg, cls.det_out_shape()),
                "side_init": cfg.REID_HEAD.INIT_WEIGHT,
                "cws": cfg.CWS,
            }
        )
        return ret

    def det_backbone(self, reid_backbone_features):
        det_res3_feat = self.side_res3(
            self.lateral_conv2(reid_backbone_features["res2"])
        )

        alpha = torch.sigmoid(self.alpha_res3)
        det_res3_feat = (
            alpha * self.lateral_conv3(reid_backbone_features["res3"])
            + (1 - alpha) * det_res3_feat
        )
        det_res4_feat = self.side_res4(det_res3_feat)

        del det_res3_feat
        alpha = torch.sigmoid(self.alpha_res4)
        det_res4_feat = (
            alpha * self.lateral_conv4(reid_backbone_features["res4"])
            + (1 - alpha) * det_res4_feat
        )
        return {"res4": det_res4_feat}


def _load_file(filename):
    from psd2.utils.file_io import PathManager
    import pickle

    if filename.endswith(".pkl"):
        with PathManager.open(filename, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        if "model" in data and "__author__" in data:
            # file is in Detectron2 model zoo format
            return data
        else:
            # assume file is from Caffe2 / Detectron1 model zoo
            if "blobs" in data:
                # Detection models have "blobs", but ImageNet models don't
                data = data["blobs"]
            data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
            return {
                "model": data,
                "__author__": "Caffe2",
                "matching_heuristics": True,
            }
    elif filename.endswith(".pyth"):
        # assume file is from pycls; no one else seems to use the ".pyth" extension
        with PathManager.open(filename, "rb") as f:
            data = torch.load(f)
        assert (
            "model_state" in data
        ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
        model_state = {
            k: v
            for k, v in data["model_state"].items()
            if not k.endswith("num_batches_tracked")
        }
        return {
            "model": model_state,
            "__author__": "pycls",
            "matching_heuristics": True,
        }

    loaded = torch.load(
        filename, map_location=torch.device("cpu")
    )  # load native pth checkpoint
    if "model" not in loaded:
        loaded = {"model": loaded}
    return loaded

class SupConLoss(nn.Module):
    #NOTE simplified
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, anchor_feature,contrast_feature, anchor_labels,contrast_labels):
        anchor_labels=anchor_labels.view(-1,1)
        contrast_labels=contrast_labels.view(-1,1)
        mask = torch.eq(anchor_labels, contrast_labels.T).float()
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
import torch.nn.functional as F
class ProtoConLoss(nn.Module):

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ProtoConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, anchor_feature,proto_feature, anchor_labels):
        anchor_labels=anchor_labels.view(-1)
        # compute logits
        anchor_logits = torch.div(
            torch.matmul(anchor_feature, proto_feature.T),
            self.temperature)
        loss=F.cross_entropy(anchor_logits,anchor_labels,reduction="none")
        # loss
        loss = (self.temperature / self.base_temperature) * loss
        loss = loss.mean()

        return loss
import numbers
import warnings
from typing import Tuple, List, Optional
from torch import Tensor
import math
class RandomErasingAfterNorm(torch.nn.Module):
    """ Randomly selects a rectangle region in an torch Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.

    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.ToTensor(),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self,pix_mean,pix_std, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__()
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")
        if isinstance(value, numbers.Number):
            value=torch.tensor([value]*pix_mean.shape[0],device=pix_mean.device)
        norm_value=(value-pix_mean)/pix_std
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = norm_value
        self.inplace = inplace

    @staticmethod
    def get_params(
            img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[List[float]] = None
    ) -> Tuple[int, int, int, int, Tensor]:
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = value[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1, )).item()
            j = torch.randint(0, img_w - w + 1, size=(1, )).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img
    @staticmethod
    def erase(img: Tensor, i: int, j: int, h: int, w: int, v: Tensor, inplace: bool = False) -> Tensor:
        """ Erase the input Tensor Image with given value.
        This transform does not support PIL Image.

        Args:
            img (Tensor Image): Tensor image of size (C, H, W) to be erased
            i (int): i in (i,j) i.e coordinates of the upper left corner.
            j (int): j in (i,j) i.e coordinates of the upper left corner.
            h (int): Height of the erased region.
            w (int): Width of the erased region.
            v: Erasing value.
            inplace(bool, optional): For in-place operations. By default is set False.

        Returns:
            Tensor Image: Erased image.
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))

        if not inplace:
            img = img.clone()

        img[..., i:i + h, j:j + w] = v
        return img

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            value = self.value
            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            return self.erase(img, x, y, h, w, v, self.inplace)
        return img