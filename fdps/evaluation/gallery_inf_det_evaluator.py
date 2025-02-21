import fdps.utils.comm as comm

import torch
import logging


from .evaluator import DatasetEvaluator
import itertools
import os
import numpy as np

from sklearn.metrics import average_precision_score
import copy
from collections import OrderedDict


from torchvision.ops import box_iou


logger = logging.getLogger(__name__)


class InfDetEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        s_threds=[0.05, 0.1, 0.2],
    ) -> None:
        self._distributed = distributed
        self._output_dir = output_dir
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.threshs = s_threds
        self.iou_threshs = [0.5, 0.7]
        self.topk = 100
        # (image name,torch concatenated [boxes, scores, reid features])
        self.inf_results = {}
        # (image name,torch concatenated [boxes, ids])
        self.gallery_gts = {}
        self.gallery_labels = {}
        self.granks_of_local_0s = []
        # det statistic
        self.y_trues = {}
        self.y_scores = {}
        self.count_gt = 0
        self.count_gt_lb = 0
        self.count_tp = 0
        self.count_tp_lb = 0


    def reset(self):
        self.inf_results = {}
        self.gallery_gts = {}
        self.granks_of_local_0s = []
        # det statistic
        self.y_trues = {iou: {k: [] for k in self.threshs} for iou in self.iou_threshs}
        self.y_scores = {iou: {k: [] for k in self.threshs} for iou in self.iou_threshs}
        self.count_gt = {iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs}
        self.count_tp = {iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs}
        self.count_gt_lb = {
            iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs
        }
        self.count_tp_lb = {
            iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs
        }

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
                    "pred_boxes": abs cx_cy_w_h boxes tensor with coordinates in org range
                    "pred_scores": scores tensor,B x N x 1
                    "reid_feats": reid features tensor
                } for one image, but batch_size dim still exists
        """

        for bi, in_dict in enumerate(inputs):
            # save gt info
            out_dict = {
                k: v[bi] for k, v in outputs.items()
            }  # one dict for each image
            gt_img_name = in_dict["image_id"]
            box_t, ids_t = torch.tensor(in_dict["org_boxes"]), torch.tensor(
                in_dict["ids"], dtype=torch.float32
            ).unsqueeze(
                1
            )  # box xyxy_abs
            gt_label = torch.cat([box_t, ids_t], dim=-1)
            self.gallery_gts[gt_img_name] = gt_label
            # save inf results
            # TODO check if topk needed
            scores_t = out_dict["pred_scores"].squeeze(1)
            if scores_t.shape[0] > self.topk:
                _, idx = torch.topk(scores_t, self.topk)
                scores_t = scores_t[idx]
                out_boxes = out_dict["pred_boxes"][idx]
                out_feats = out_dict["reid_feats"][idx]
            else:
                out_boxes = out_dict["pred_boxes"]
                out_feats = out_dict["reid_feats"]
            inf_rt = torch.cat([out_boxes, scores_t[:, None], out_feats], dim=-1)
            self.inf_results[gt_img_name] = inf_rt.to(self._cpu_device)
        for st in self.threshs:
            self._det_proc(inputs, outputs, st)

    def _det_proc(self, inputs, outputs, sthred):
            for bi, in_dict in enumerate(inputs):
                out_dict = {
                    k: v[bi] for k, v in outputs.items()
                }  # one dict for each image
                gt_boxes = torch.tensor(in_dict["org_boxes"])
                gt_pids = torch.tensor(in_dict["ids"], dtype=torch.long)
                det_boxes = out_dict["pred_boxes"].cpu()
                det_scores = out_dict["pred_scores"].cpu()
                num_gts = gt_boxes.shape[0]
                num_lb_gts = (gt_pids > -1).sum().item()
                if det_boxes.shape[0] > 0:
                    if det_boxes.shape[0] > self.topk:
                        topk_scores, topk_idxs = torch.topk(det_scores[:, 0], self.topk)
                        topk_det_boxes = det_boxes[topk_idxs]
                        det_scores = topk_scores[topk_scores >= sthred]
                        det_boxes = topk_det_boxes[topk_scores >= sthred]
                        num_dets = det_boxes.shape[0]
                    else:
                        det_scores_mask = det_scores[:, 0] >= sthred
                        det_boxes = det_boxes[det_scores_mask]
                        num_dets = det_boxes.shape[0]
                else:
                    num_dets = 0
                if num_dets == 0:
                    for iou_t in self.iou_threshs:
                        self.count_gt[iou_t][sthred] += num_gts
                        self.count_gt_lb[iou_t][sthred] += num_lb_gts
                    continue
                ious = box_iou(gt_boxes, det_boxes)
                for iou_t in self.iou_threshs:
                    tfmat = ious >= iou_t
                    # for each det, keep only the largest iou of all the gt
                    for j in range(num_dets):
                        largest_ind = torch.argmax(ious[:, j])
                        for i in range(num_gts):
                            if i != largest_ind:
                                tfmat[i, j] = False
                    # for each gt, keep only the largest iou of all the det
                    for i in range(num_gts):
                        largest_ind = np.argmax(ious[i, :])
                        for j in range(num_dets):
                            if j != largest_ind:
                                tfmat[i, j] = False
                    for j in range(num_dets):
                        self.y_scores[iou_t][sthred].append(det_scores[j].item())
                        self.y_trues[iou_t][sthred].append(tfmat[:, j].any())
                    self.count_tp[iou_t][sthred] += tfmat.sum().item()
                    self.count_gt[iou_t][sthred] += num_gts
                    tfmat_lb = tfmat[gt_pids > -1, :]
                    self.count_tp_lb[iou_t][sthred] += tfmat_lb.sum().item()
                    self.count_gt_lb[iou_t][sthred] += num_lb_gts

    def evaluate(self):
        if self._distributed:
            if comm.get_local_rank() == 0:
                self.granks_of_local_0s.append(comm.get_rank())
            comm.synchronize()
            save_ranks = comm.all_gather(self.granks_of_local_0s)
            save_ranks = list(set(itertools.chain(*save_ranks)))
            save_rts = {}
            save_gts = {}
            for rk in save_ranks:
                g_gts = comm.gather(self.gallery_gts, dst=rk)
                inf_rts = comm.gather(self.inf_results, dst=rk)
                if len(inf_rts) > 0:
                    for rts in inf_rts:
                        save_rts.update(rts)
                if len(g_gts) > 0:
                    for gts in g_gts:
                        save_gts.update(gts)

            y_true_rs = comm.gather(self.y_trues, dst=0)
            y_score_rs = comm.gather(self.y_scores, dst=0)
            count_gt_rs = comm.gather(self.count_gt, dst=0)
            count_tp_rs = comm.gather(self.count_tp, dst=0)
            count_gt_lb_rs = comm.gather(self.count_gt_lb, dst=0)
            count_tp_lb_rs = comm.gather(self.count_tp_lb, dst=0)
            y_trues = {iou: {k: [] for k in self.threshs} for iou in self.iou_threshs}
            y_scores = {iou: {k: [] for k in self.threshs} for iou in self.iou_threshs}
            count_gts = {iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs}
            count_tps = {iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs}
            count_gts_lb = {
                iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs
            }
            count_tps_lb = {
                iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs
            }
            for iou_t in self.iou_threshs:
                for st in self.threshs:
                    y_trues[iou_t][st] = list(
                        itertools.chain(*[yt[iou_t][st] for yt in y_true_rs])
                    )
                    y_scores[iou_t][st] = list(
                        itertools.chain(*[ys[iou_t][st] for ys in y_score_rs])
                    )
                    count_gts[iou_t][st] = sum([cg[iou_t][st] for cg in count_gt_rs])
                    count_tps[iou_t][st] = sum([ct[iou_t][st] for ct in count_tp_rs])
                    count_gts_lb[iou_t][st] = sum(
                        [cg[iou_t][st] for cg in count_gt_lb_rs]
                    )
                    count_tps_lb[iou_t][st] = sum(
                        [ct[iou_t][st] for ct in count_tp_lb_rs]
                    )
            if len(save_rts) == 0 or len(save_gts) == 0 :
                comm.synchronize()
                return {}

        else:
            save_rts = self.inf_results
            save_gts = self.gallery_gts
            y_true = self.y_trues
            y_score = self.y_scores
            count_gts = self.count_gt
            count_tps = self.count_tp
            count_gts_lb = self.count_gt_lb
            count_tps_lb = self.count_tp_lb
        """save_gts = {k: v.to(self._cpu_device) for k, v in save_gts.items()}
        save_rts = {k: v.to(self._cpu_device) for k, v in save_rts.items()}"""
        save_dict = {"gts": save_gts, "infs": save_rts}
        save_path = os.path.join(self._output_dir, "_gallery_gt_inf.pt")
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        torch.save(save_dict, save_path)
        comm.synchronize()
        # det eval
        if not comm.is_main_process():
            return {}
        det_result = OrderedDict()
        det_result["detection"] = {}
        for iou_t in self.iou_threshs:
            for st in self.threshs:
                count_tp = count_tps[iou_t][st]
                count_gt = count_gts[iou_t][st]
                count_tp_lb = count_tps_lb[iou_t][st]
                count_gt_lb = count_gts_lb[iou_t][st]
                y_true = y_trues[iou_t][st]
                y_score = y_scores[iou_t][st]
                det_rate = count_tp * 1.0 / count_gt
                det_rate_lb = count_tp_lb * 1.0 / count_gt_lb
                if count_tp == 0:
                    det_result["detection"].update(
                        {"AP_{}".format(st): 0, "Recall_{}".format(st): 0}
                    )
                    continue
                y_true = np.array(y_true, dtype=np.int32)
                y_score = np.array(y_score, dtype=np.float32)
                ap = average_precision_score(y_true, y_score) * det_rate
                det_result["detection"].update(
                    {
                        "AP_iou{}_score{}".format(iou_t, st): ap,
                        "Recall_iou{}_score{}".format(iou_t, st): det_rate,
                        "RecallLb_iou{}_score{}".format(iou_t, st): det_rate_lb,
                    }
                )

        return copy.deepcopy(det_result)

