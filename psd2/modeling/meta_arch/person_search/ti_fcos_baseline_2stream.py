from psd2.structures.boxes import Boxes
from ..build import META_ARCH_REGISTRY
import torch
from psd2.utils.events import get_event_storage
from psd2.config import configurable
import torch.nn as nn
from psd2.layers.pooling import *
import torch.utils.checkpoint as checkpoint
import copy
import time
from psd2.modeling.meta_arch.person_search import TiFcos_Baseline,TiFcos_Baseline_JointDc

from psd2.modeling.backbone.resnet import (
    ResNet,
    BottleneckBlock,
    DeformBottleneckBlock,
)
import torch.nn.init as init
import torch.nn.functional as tF
import copy


@META_ARCH_REGISTRY.register()
class TiFcos_2StreamOnCrops(TiFcos_Baseline):
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
        raise NotImplementedError
    def forward_ps(self, det_pred, det_backbone_features, image_list, gts):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        # print("reid backbone time: {}".format(time.time()-t0))
        if self.training:
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples(det_pred, gts)

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
                    if isinstance(pos_featmaps,torch.Tensor):
                        vis_feat=pos_featmaps
                    else:
                        vis_feat=pos_featmaps[-1]
                    self.visualize_training_ps(
                        image_list.tensor,
                        vis_feat.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(pos_featmaps)
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            losses= self.reid_loss(pos_embs, pos_ids, None)
            crops=self.get_crops(image_list,pos_boxes)
            crop_embs=self.get_crop_embs(crops)
            crop_ids=pos_ids.clone()
            losses.update(self.crop_img_loss(pos_embs,pos_ids,crop_embs,crop_ids))
            return losses
        else:
            t0=time.time()
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
            # print("reid pred time: {}".format(time.time()-t0))
            return p_embs

    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        raise NotImplementedError

@META_ARCH_REGISTRY.register()
class TiFcos_JointDc_2StreamOnCrops(TiFcos_2StreamOnCrops,TiFcos_Baseline_JointDc):
    def forward(self, input_list):
        return TiFcos_Baseline_JointDc.forward(self,input_list)
    def forward_det(self, image_list, features, gt_instances):
        return TiFcos_Baseline_JointDc.forward_det(self, image_list, features, gt_instances)
    def get_ps_pos_samples(self,gts):
        return TiFcos_Baseline_JointDc.get_ps_pos_samples(self,gts)
    def forward_ps(self, det_backbone_features, image_list, gts,det_pred=None):
        reid_bk_feats = self.get_reid_backbone_features(
            det_backbone_features, image_list
        )
        # print("reid backbone time: {}".format(time.time()-t0))
        if self.training:
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples( gts)

            pos_featmaps = self.get_reid_person_features(
                reid_bk_feats, pos_boxes
            )  # nf x c x h x w

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    if isinstance(pos_featmaps,torch.Tensor):
                        vis_feat=pos_featmaps
                    else:
                        vis_feat=pos_featmaps[-1]
                    self.visualize_training_ps(
                        image_list.tensor,
                        vis_feat.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(pos_featmaps)
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            losses= self.reid_loss(pos_embs, pos_ids, None)
            crops=self.get_crops(image_list,pos_boxes)
            crop_embs=self.get_crop_embs(crops)
            crop_ids=pos_ids.clone()
            losses.update(self.crop_img_loss(pos_embs,pos_ids,crop_embs,crop_ids))
            return losses
        else:
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
            # print("reid pred time: {}".format(time.time()-t0))
            return p_embs

@META_ARCH_REGISTRY.register()
class TiFcos_C4Side2StreamOnCrops(TiFcos_2StreamOnCrops):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super().__init__(**kwargs)

        self.side_init = side_init
        self.alpha_res3 = nn.Parameter(torch.tensor(0.0))
        self.alpha_res4 = nn.Parameter(torch.tensor(0.0))
        # side reid network
        self.bn_neck = bn_neck
        self.side_res3 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 4,
                    "stride_per_block": [2, 1, 1, 1],
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
                    "num_blocks": 6,
                    "stride_per_block": [2, 1, 1, 1, 1, 1],
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
        self.side_res5 = nn.Sequential(
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
        # freeze det params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.anchor_generator.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg):
        res = super().from_config(cfg)
        bn_neck = nn.BatchNorm1d(cfg.REID_HEAD.PERSON_FEATURE.DIM)
        init.constant_(bn_neck.bias, 0.0)
        init.normal_(bn_neck.weight, std=0.01)
        bn_neck.bias.requires_grad_(False)
        res["bn_neck"] = bn_neck
        res["side_init"] = cfg.REID_HEAD.INIT_WEIGHT
        return res
    def get_crop_embs(self,crops):
        crop_bk_feats=self.backbone.bottom_up(crops)
        crop_feats=self.get_reid_backbone_features(crop_bk_feats,None)
        crop_feats=(
                    checkpoint.checkpoint(self.side_res5, crop_feats)
                    if self.use_checkpoint and self.training
                    else self.side_res5(crop_feats)
                )
        crop_feats=self.get_reid_embed(crop_feats)
        return crop_feats

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
            bn_neck_params = {}
            for si in range(3):
                res_name = "res{}".format(si + 3)
                for pn, pa in state_dict.items():
                    if res_name in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        side_res_params[si][pn.split(res_name)[1][1:]] = pa
                    elif "bn_neck" in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        bn_neck_params[pn.split("bn_neck")[1][1:]] = pa
            self.side_res3.load_state_dict(side_res_params[0],strict=False)
            self.side_res4.load_state_dict(side_res_params[1],strict=False)
            self.side_res5.load_state_dict(side_res_params[2],strict=False)
            if len(bn_neck_params) > 0:
                self.bn_neck.load_state_dict(bn_neck_params)
        return output

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        reid_res3_feat = (
            checkpoint.checkpoint(self.side_res3, det_backbone_features["res2"])
            if self.use_checkpoint
            else self.side_res3(det_backbone_features["res2"])
        )
        alpha = torch.sigmoid(self.alpha_res3)
        reid_res3_feat = (
            alpha * det_backbone_features["res3"] + (1 - alpha) * reid_res3_feat
        )
        reid_res4_feat = (
            checkpoint.checkpoint(self.side_res4, reid_res3_feat)
            if self.use_checkpoint
            else self.side_res4(reid_res3_feat)
        )
        del reid_res3_feat
        alpha = torch.sigmoid(self.alpha_res4)
        reid_res4_feat = (
            alpha * det_backbone_features["res4"] + (1 - alpha) * reid_res4_feat
        )
        return reid_res4_feat

    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats = self.reid_pooler([reid_backbone_features], d2_boxes)
        roi_feats = (
            checkpoint.checkpoint(self.side_res5, roi_feats)
            if self.use_checkpoint and self.training
            else self.side_res5(roi_feats)
        )  # n x c x h x w
        return roi_feats

    def get_reid_embed(self, p_feat_maps):
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        return tF.normalize(pfeat_embs, dim=-1)

@META_ARCH_REGISTRY.register()
class TiFcos_C4Side2StreamOnCrops_JointDc(TiFcos_JointDc_2StreamOnCrops,TiFcos_C4Side2StreamOnCrops):
    @configurable
    def __init__(self,bn_neck, side_init,*args, **kwargs) -> None:
        TiFcos_JointDc_2StreamOnCrops.__init__(self,*args,**kwargs)
        self.side_init = side_init
        self.alpha_res3 = nn.Parameter(torch.tensor(0.0))
        self.alpha_res4 = nn.Parameter(torch.tensor(0.0))
        # side reid network
        self.bn_neck = bn_neck
        self.side_res3 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 4,
                    "stride_per_block": [2, 1, 1, 1],
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
                    "num_blocks": 6,
                    "stride_per_block": [2, 1, 1, 1, 1, 1],
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
        self.side_res5 = nn.Sequential(
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
    
    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        reid_res3_feat = (
            checkpoint.checkpoint(self.side_res3, det_backbone_features["res2"].detach())
            if self.use_checkpoint
            else self.side_res3(det_backbone_features["res2"].detach())
        )
        alpha = torch.sigmoid(self.alpha_res3)
        reid_res3_feat = (
            alpha * det_backbone_features["res3"].detach() + (1 - alpha) * reid_res3_feat
        )
        reid_res4_feat = (
            checkpoint.checkpoint(self.side_res4, reid_res3_feat)
            if self.use_checkpoint
            else self.side_res4(reid_res3_feat)
        )
        del reid_res3_feat
        alpha = torch.sigmoid(self.alpha_res4)
        reid_res4_feat = (
            alpha * det_backbone_features["res4"].detach() + (1 - alpha) * reid_res4_feat
        )
        return reid_res4_feat

@META_ARCH_REGISTRY.register()
class TiFcos_C4Side2StreamOnCropsOimProtoCon(TiFcos_C4Side2StreamOnCrops):
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
class TiFcos_C4Side2StreamOnCropsOimProtoConRe(TiFcos_C4Side2StreamOnCropsOimProtoCon):
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
class TiFcos_C4Side2StreamOnCropsOimProtoConReFlip(TiFcos_C4Side2StreamOnCropsOimProtoConRe):
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
class TiFcos_C4Side2StreamOnCropsOimProtoConReFlip_JointDc(TiFcos_C4Side2StreamOnCrops_JointDc):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.re=RandomErasingAfterNorm(self.pixel_mean[:,0,0],self.pixel_std[:,0,0])
        self.crop_loss=copy.deepcopy(self.reid_loss)
        self.contrast_loss=ProtoConLoss()
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
class TiFcos_C4Side2StreamOnCropsOimProtoConReFlip_JointC(TiFcos_C4Side2StreamOnCropsOimProtoConReFlip_JointDc):
    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        reid_res4_feat = det_backbone_features["res4"]
        return reid_res4_feat
    def get_crop_embs(self,crops):
        crop_bk_feats=self.backbone.bottom_up(crops)
        crop_feats=self.get_reid_backbone_features(crop_bk_feats,None)
        crop_feats=(
                    checkpoint.checkpoint(self.backbone.bottom_up.res5, crop_feats)
                    if self.use_checkpoint and self.training
                    else self.backbone.bottom_up.res5(crop_feats)
                )
        crop_feats=self.get_reid_embed(crop_feats)
        return crop_feats
    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats = self.reid_pooler([reid_backbone_features], d2_boxes)
        roi_feats = (
            checkpoint.checkpoint(self.backbone.bottom_up.res5, roi_feats)
            if self.use_checkpoint and self.training
            else self.backbone.bottom_up.res5(roi_feats)
        )  # n x c x h x w
        return roi_feats

@META_ARCH_REGISTRY.register()
class TiFcos_C4SideD2StreamOnCropsOimProtoConReFlip(TiFcos_C4Side2StreamOnCropsOimProtoConReFlip):
    @configurable
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.side_res3 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 4,
                    "stride_per_block": [2, 1, 1, 1],
                    "in_channels": 256,
                    "out_channels": 512,
                    "norm": "BN",
                    "bottleneck_channels": 128,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": DeformBottleneckBlock,
                    "deform_modulated": True,
                    "deform_num_groups": 1
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
                    "norm": "BN",
                    "bottleneck_channels": 256,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": DeformBottleneckBlock,
                    "deform_modulated": True,
                    "deform_num_groups": 1
                }
            )
        )
        self.side_res5 = nn.Sequential(
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
                    "block_class": DeformBottleneckBlock,
                    "deform_modulated": True,
                    "deform_num_groups": 1
                }
            )
        )



from psd2.layers.metric_loss import TripletLoss
@META_ARCH_REGISTRY.register()
class TiFcos_C4Side2StreamOnCropsOimProtoConReFlipTrip(TiFcos_C4Side2StreamOnCropsOimProtoConReFlip):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.triplet_loss = TripletLoss(0.3, "mean")
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        losses=super().crop_img_loss(img_embs,img_pids,crop_embs,crop_pids)
        # img trip
        oim_lookup = self.reid_loss.lb_layer.lookup_table
        lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=img_pids.dtype, device=img_pids.device
            )
        feats1 = img_embs[img_pids > -1]
        feats2 = torch.cat([img_embs[img_pids > -1], oim_lookup], dim=0)
        if feats1.shape[0] < 1:
            losses["loss_triplet"] = torch.zeros(1, device=self.device)
        else:
            trip = self.triplet_loss(
                    feats1,
                    feats2,
                    img_pids[img_pids > -1],
                    torch.cat([img_pids[img_pids > -1], lookup_ids], dim=0),
                    normalize_feature=True,
                )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k] = v*self.w_kd
        # crop trip
        oim_lookup = self.crop_loss.lb_layer.lookup_table
        lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=crop_pids.dtype, device=crop_pids.device
            )
        feats1 = crop_embs[crop_pids > -1]
        feats2 = torch.cat([crop_embs[crop_pids > -1], oim_lookup], dim=0)
        if feats1.shape[0] < 1:
            losses["loss_triplet_crop"] = torch.zeros(1, device=self.device)
        else:
            trip = self.triplet_loss(
                    feats1,
                    feats2,
                    crop_pids[crop_pids > -1],
                    torch.cat([crop_pids[crop_pids > -1], lookup_ids], dim=0),
                    normalize_feature=True,
                )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k+"_crop"] = v*self.w_kd
        return losses

@META_ARCH_REGISTRY.register()
class TiFcos_C4SideD2StreamOnCropsOimProtoConReFlipTrip(TiFcos_C4SideD2StreamOnCropsOimProtoConReFlip):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.triplet_loss = TripletLoss(0.3, "mean")
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        losses=super().crop_img_loss(img_embs,img_pids,crop_embs,crop_pids)
        # img trip
        oim_lookup = self.reid_loss.lb_layer.lookup_table
        lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=img_pids.dtype, device=img_pids.device
            )
        feats1 = img_embs[img_pids > -1]
        feats2 = torch.cat([img_embs[img_pids > -1], oim_lookup], dim=0)
        if feats1.shape[0] < 1:
            losses["loss_triplet"] = torch.zeros(1, device=self.device)
        else:
            trip = self.triplet_loss(
                    feats1,
                    feats2,
                    img_pids[img_pids > -1],
                    torch.cat([img_pids[img_pids > -1], lookup_ids], dim=0),
                    normalize_feature=True,
                )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k] = v*self.w_kd
        # crop trip
        oim_lookup = self.crop_loss.lb_layer.lookup_table
        lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=crop_pids.dtype, device=crop_pids.device
            )
        feats1 = crop_embs[crop_pids > -1]
        feats2 = torch.cat([crop_embs[crop_pids > -1], oim_lookup], dim=0)
        if feats1.shape[0] < 1:
            losses["loss_triplet_crop"] = torch.zeros(1, device=self.device)
        else:
            trip = self.triplet_loss(
                    feats1,
                    feats2,
                    crop_pids[crop_pids > -1],
                    torch.cat([crop_pids[crop_pids > -1], lookup_ids], dim=0),
                    normalize_feature=True,
                )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k+"_crop"] = v*self.w_kd
        return losses



@META_ARCH_REGISTRY.register()
class TiFcos_C4Side2StreamOnCropsOimProtoConReFlipTrip_JointDc(TiFcos_C4Side2StreamOnCropsOimProtoConReFlip_JointDc):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.triplet_loss = TripletLoss(0.3, "mean")
    def crop_img_loss(self,img_embs,img_pids,crop_embs,crop_pids):
        losses=super().crop_img_loss(img_embs,img_pids,crop_embs,crop_pids)
        # img trip
        oim_lookup = self.reid_loss.lb_layer.lookup_table
        lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=img_pids.dtype, device=img_pids.device
            )
        feats1 = img_embs[img_pids > -1]
        feats2 = torch.cat([img_embs[img_pids > -1], oim_lookup], dim=0)
        if feats1.shape[0] < 1:
            losses["loss_triplet"] = torch.zeros(1, device=self.device)
        else:
            trip = self.triplet_loss(
                    feats1,
                    feats2,
                    img_pids[img_pids > -1],
                    torch.cat([img_pids[img_pids > -1], lookup_ids], dim=0),
                    normalize_feature=True,
                )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k] = v*self.w_kd
        # crop trip
        oim_lookup = self.crop_loss.lb_layer.lookup_table
        lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=crop_pids.dtype, device=crop_pids.device
            )
        feats1 = crop_embs[crop_pids > -1]
        feats2 = torch.cat([crop_embs[crop_pids > -1], oim_lookup], dim=0)
        if feats1.shape[0] < 1:
            losses["loss_triplet_crop"] = torch.zeros(1, device=self.device)
        else:
            trip = self.triplet_loss(
                    feats1,
                    feats2,
                    crop_pids[crop_pids > -1],
                    torch.cat([crop_pids[crop_pids > -1], lookup_ids], dim=0),
                    normalize_feature=True,
                )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k+"_crop"] = v*self.w_kd
        return losses


# TODO
@META_ARCH_REGISTRY.register()
class TiFcos_NextC4Side2StreamOnCropsOimProtoConReFlip(TiFcos_C4Side2StreamOnCropsOimProtoConReFlip):
    @configurable
    def __init__(
        self,
        conv_next,**kwargs,) -> None:
        super().__init__(**kwargs)
        # next stage 2
        self.side_res3 = nn.Sequential(copy.deepcopy(conv_next.downsample_layers[1]),copy.deepcopy(conv_next.stages[1]))
        
        self.side_res4 = nn.Sequential(copy.deepcopy(conv_next.downsample_layers[2]),copy.deepcopy(conv_next.stages[2]))
        self.side_res5 = nn.Sequential(copy.deepcopy(conv_next.downsample_layers[3]),copy.deepcopy(conv_next.stages[3]),copy.deepcopy(conv_next.norm3))
        del conv_next
        # remove last stride 
        self.side_res5[0][1].stride=(1,1)
        self.side_res5[0][1].padding=(1,1)
        self.side_res5[0][1].dilation=(2,2)

    def load_state_dict(self, *args, **kws):
        # NOTE side init is not needed
        output = super(TiFcos_Baseline,self).load_state_dict(*args, **kws)
        return output
    @classmethod
    def from_config(cls, cfg):
        raise NotImplementedError
        ret = super().from_config(cfg)
        next_res5=nn.Sequential(copy.deepcopy(ret["backbone"].downsample_layers[3]),copy.deepcopy(ret["backbone"].stages[3]),copy.deepcopy(ret["backbone"].norm3))
        bk=ret["backbone"]
        # roi_heads=OuterRes5ROIHeads(cfg, bk.output_shape(),next_res5,768)
        ret["roi_heads"]=roi_heads
        conv_next=convnext_tiny(cfg,None)
        ret["conv_next"]=conv_next
        return ret
    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        reid_res3_feat = (
            checkpoint.checkpoint(self.side_res3, det_backbone_features["stage1_unorm"])
            if self.use_checkpoint
            else self.side_res3(det_backbone_features["stage1_unorm"])
        )
        alpha = torch.sigmoid(self.alpha_res3)
        reid_res3_feat = (
            alpha * det_backbone_features["stage2_unorm"] + (1 - alpha) * reid_res3_feat
        )
        reid_res4_feat = (
            checkpoint.checkpoint(self.side_res4, reid_res3_feat)
            if self.use_checkpoint
            else self.side_res4(reid_res3_feat)
        )
        del reid_res3_feat
        alpha = torch.sigmoid(self.alpha_res4)
        reid_res4_feat = (
            alpha * det_backbone_features["stage3_unorm"] + (1 - alpha) * reid_res4_feat
        )
        return reid_res4_feat

from psd2.modeling.backbone import convnext_tiny
@META_ARCH_REGISTRY.register()
class TiFcos_C4DSide2StreamOnCropsOimProtoConReFlip(TiFcos_C4Side2StreamOnCropsOimProtoConReFlip):
    @configurable
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.side_res3 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 4,
                    "stride_per_block": [2, 1, 1, 1],
                    "in_channels": 256,
                    "out_channels": 512,
                    "norm": "BN",
                    "bottleneck_channels": 128,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": DeformBottleneckBlock,
                    "deform_modulated": True,
                    "deform_num_groups": 1
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
                    "norm": "BN",
                    "bottleneck_channels": 256,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": DeformBottleneckBlock,
                    "deform_modulated": True,
                    "deform_num_groups": 1
                }
            )
        )
        self.side_res5 = nn.Sequential(
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
                    "block_class": DeformBottleneckBlock,
                    "deform_modulated": True,
                    "deform_num_groups": 1
                }
            )
        )

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