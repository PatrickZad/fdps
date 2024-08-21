
import psd2.utils.comm as comm
import torch
import logging
from .evaluator import DatasetEvaluator
import os
import numpy as np
import cv2
import copy
from torchvision.ops import box_iou,roi_align
import shutil
from psd2.utils.visualizer import mlvl_pca_feat
import torch.nn.functional as tF
logger = logging.getLogger(__name__)


class InfVisEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
    ) -> None:
        self._distributed = distributed
        self._output_dir = "/".join(["visualization"] + output_dir.split("/")[1:])
        self.emb_dir = os.path.join(self._output_dir, "embs")
        self.ifeat_dir = os.path.join(self._output_dir, "ifeats")
        self.pfeat_dir = os.path.join(self._output_dir, "pfeats")
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.embs=[]
        self.emb_pids=[]
        lrk = comm.get_local_rank()
        if lrk == 0:
            for vdir in [self.emb_dir,self.ifeat_dir,self.pfeat_dir]:
                if not os.path.exists(vdir):
                    os.makedirs(vdir)

    def reset(self):
        self.embs=[]
        self.emb_pids=[]
    def vis_output(self,img_t,embeddins,emb_boxes,gt_pids,gt_boxes,ifeat):
        img_mean=torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        img_std=torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_rgbt=(img_t*img_std+img_mean)*255.0 # c x h x w
        # save embeddings and match pids
        emb_ids=[]
        if emb_boxes.shape[0]>0:
            ious = box_iou(gt_boxes, emb_boxes)
            match_mat=ious>0.5
            for j in range(emb_boxes.shape[0]):
                largest_ind = torch.argmax(ious[:, j])
                for i in range(gt_boxes.shape[0]):
                    if i != largest_ind:
                        match_mat[i, j] = False
            for i in range(gt_boxes.shape[0]):
                largest_ind = torch.argmax(ious[i, :])
                for j in range(emb_boxes.shape[0]):
                    if j != largest_ind:
                        match_mat[i, j] = False
            match_mat=match_mat.int()
            for j in range(emb_boxes.shape[0]):
                match_ind = torch.argmax(match_mat[:, j])
                if match_mat[match_ind,j]==1:
                    emb_ids.append(gt_pids[match_ind])
                else:
                    emb_ids.append(-2)
        ifeat_vis,pvis=None,None
        try:
            # image feature pca and attn, cat with org
            img_h,img_w=img_rgbt.shape[1:]
            #feat_pca=mlvl_pca_feat([ifeat[None]])[0]
            #feat_pca=tF.interpolate(feat_pca,(img_h,img_w),mode="bilinear",align_corners=True)[0]
            # replace pca by global attn
            feat_pca=(ifeat**2).sum(0)
            feat_pca=255*(feat_pca-feat_pca.min())/(feat_pca.max() - feat_pca.min() + 1e-12)
            feat_pca=feat_pca.cpu().numpy().astype(np.uint8) # hw
            feat_pca=cv2.applyColorMap(feat_pca, cv2.COLORMAP_JET)  # bgr hwc
            feat_pca=torch.tensor(feat_pca[...,::-1].copy(),dtype=torch.float32).permute(2, 0, 1) # rgb chw
            feat_pca=tF.interpolate(feat_pca[None],(img_h,img_w),mode="bilinear",align_corners=True)[0]
            feat_pca=img_rgbt*0.6+feat_pca*0.4

            # NOTE remove background attn
            feat_attn=torch.zeros(img_h,img_w)
            tgt_size=[256,128*3]
            p_vis=[]
            for box_if in emb_boxes: # xyxy
                box_if[0]=torch.clamp(box_if[0],min=0,max=img_w)
                box_if[2]=torch.clamp(box_if[2],min=0,max=img_w)
                box_if[1]=torch.clamp(box_if[1],min=0,max=img_h)
                box_if[3]=torch.clamp(box_if[3],min=0,max=img_h)
                box_i=box_if.long()
                p_img=img_rgbt[:,box_i[1]:box_i[3], box_i[0]:box_i[2]]
                p_pca=feat_pca[:,box_i[1]:box_i[3], box_i[0]:box_i[2]]
                p_h,p_w=box_i[3]-box_i[1],box_i[2]-box_i[0]
                pfeat=roi_align(ifeat[None].cuda(),[box_if[None].cuda()],(p_h,p_w),1/16.,aligned=True).cpu()[0]
                p_attn=(pfeat**2).sum(0)
                p_attn=255*(p_attn-p_attn.min())/(p_attn.max() - p_attn.min() + 1e-12)
                feat_attn[box_i[1]:box_i[3], box_i[0]:box_i[2]]=p_attn
                p_attn_arr=p_attn.cpu().numpy().astype(np.uint8) # hw
                p_attn_arr=cv2.applyColorMap(p_attn_arr, cv2.COLORMAP_JET)  # bgr hwc
                p_attn_arr=torch.tensor(p_attn_arr[...,::-1].copy(),dtype=torch.float32).permute(2, 0, 1) # rgb chw
                p_attn_arr=p_img*0.6+p_attn_arr*0.4
                p_v=tF.interpolate(torch.cat([p_img,p_pca,p_attn_arr],dim=-1)[None],size=tgt_size,mode="bilinear",align_corners=True)[0]
                p_vis.append(p_v)
            feat_attn=feat_attn.cpu().numpy().astype(np.uint8) # hw
            feat_attn=cv2.applyColorMap(feat_attn, cv2.COLORMAP_JET)  # bgr hwc
            feat_attn=torch.tensor(feat_attn[...,::-1].copy(),dtype=torch.float32).permute(2, 0, 1) # rgb chw
            feat_attn=tF.interpolate(feat_attn[None],(img_h,img_w),mode="bilinear",align_corners=True)[0]
            feat_attn=img_rgbt*0.6+feat_attn*0.4
            ifeat_vis=torch.cat([img_rgbt,feat_pca,feat_attn],dim=-1)
            # person feature pca and attn, cat with org
            pvis=torch.cat(p_vis,dim=1)
        except Exception as e:
            print(e)
        
        return embeddins,emb_ids,ifeat_vis,pvis




    def process(self, inputs, outputs):
        """
        Args:
            inputs:
                a list of
                {
                    "file_name": image paths,
                    "image_id": image name,
                    "image": image tensor,
                    "width": image width,
                    "height": image height,
                    "boxes": boxes array,
                    "ids": person ids list,
                    "org_width": original width,
                    "org_height": original height
                }
            outputs:
                {
                    "p_embs": 
                    "p_boxes": 
                    "i_feat":
                } for one image, w/o batch_size dim 
        """
        for i,input_i in enumerate(inputs):
            img_t=input_i["image"].cpu()
            img_name=input_i["image_id"]
            gt_boxes=torch.tensor(input_i["boxes"])
            gt_pids=input_i["ids"]
            embeddins,emb_ids,ifeat_vis,pvis=self.vis_output(img_t,outputs[i]["p_embs"],outputs[i]["p_boxes"],gt_pids,gt_boxes,outputs[i]["i_feat"])
            self.embs.extend(embeddins)
            self.emb_pids.extend(emb_ids)
            if ifeat_vis is not None:
                ifeat_arr=ifeat_vis.permute(1,2,0).numpy()[...,::-1]
                cv2.imwrite(os.path.join(self.ifeat_dir,img_name),ifeat_arr)
            if pvis is not None:
                pfeat_arr=pvis.permute(1,2,0).numpy()[...,::-1]
                cv2.imwrite(os.path.join(self.pfeat_dir,img_name),pfeat_arr)

    

    def evaluate(self):
        if self._distributed:

            comm.synchronize()
            save_embs=self.embs
            save_pids=self.emb_pids
            save_data=(save_embs,save_pids)
            save_all = comm.gather(save_data,dst=0)
            if comm.get_local_rank()!=0:
                comm.synchronize()
            else:
                embs_all=[]
                emb_ids_all=[]
                for data in save_all:
                    embs_all.extend(data[0])
                    emb_ids_all.extend[data[1]]
                torch.save({"p_embs":embs_all,"p_ids":emb_ids_all},os.path.join(self.emb_dir,"p_embs.pth"))
                comm.synchronize()
        else:
            torch.save({"p_embs":self.embs,"p_ids":self.emb_pids},os.path.join(self.emb_dir,"p_embs.pth"))
        vis_result = {}
        return copy.deepcopy(vis_result)

    