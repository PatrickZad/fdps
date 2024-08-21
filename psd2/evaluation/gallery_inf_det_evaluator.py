from turtle import ycor
from PIL import Image
import psd2.utils.comm as comm

import torch
import logging

from psd2.utils.visualizer import Visualizer
from .evaluator import DatasetEvaluator
import itertools
import os
import numpy as np

from sklearn.metrics import average_precision_score
import copy
from collections import OrderedDict

from os.path import join as opj
from torchvision.ops import box_iou
import shutil
from psd2.utils.visualizer import domain_hist_img
import time

logger = logging.getLogger(__name__)


class InfDetEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        s_threds=[0.05, 0.1, 0.2],
        vis=False,
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
        # det score vis to tb, for incmt spps
        self.det_scores_per_domain = []
        # make vis dirs
        vis_dir = opj(self._output_dir, dataset_name, "vis", "det")
        self.svis_dirs = {iou_t: {} for iou_t in self.iou_threshs}
        for iou_t in self.iou_threshs:
            for scr in s_threds:
                svis_dir = opj(vis_dir, str(iou_t), str(scr))
                self.svis_dirs[iou_t][scr] = svis_dir
        if not vis:
            self._vis_samp = _trivial_vis
            self._vis_domain = _trivial_vis
            return

        lrk = comm.get_local_rank()
        if lrk == 0:
            if os.path.exists(vis_dir):
                try:
                    shutil.rmtree(os.path.realpath(vis_dir))
                except Exception as e:
                    logger.info(str(e) + " occures when deleting files !")
        for iou_t in self.iou_threshs:
            for scr in s_threds:
                svis_dir = opj(vis_dir, str(iou_t), str(scr))
                svis_dir_score = opj(svis_dir, "scores")
                if lrk == 0:
                    if not os.path.exists(svis_dir):
                        os.makedirs(svis_dir)
                        os.makedirs(svis_dir_score)

    def reset(self):
        self.inf_results = {}
        self.gallery_gts = {}
        self.granks_of_local_0s = []
        self.gallery_fnames = {}  # add for vis
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
        self.det_scores_per_domain = []

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
        if "pred_scores_domain" in outputs:
            self.det_scores_per_domain.append(
                [scores.reshape(-1).cpu() for scores in outputs["pred_scores_domain"]]
            )

        for bi, in_dict in enumerate(inputs):
            # save gt info
            out_dict = {
                k: v[bi] for k, v in outputs.items() if k != "pred_scores_domain"
            }  # one dict for each image
            gt_img_name = in_dict["image_id"]
            box_t, ids_t = torch.tensor(in_dict["org_boxes"]), torch.tensor(
                in_dict["ids"], dtype=torch.float32
            ).unsqueeze(
                1
            )  # box xyxy_abs
            gt_label = torch.cat([box_t, ids_t], dim=-1)
            self.gallery_gts[gt_img_name] = gt_label
            self.gallery_fnames[gt_img_name] = in_dict["file_name"]
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
        if "pred_scores_domain" in outputs:
            for bi, in_dict in enumerate(inputs):
                scores_domain = [
                    scores[bi].cpu().reshape(-1)
                    for scores in outputs["pred_scores_domain"]
                ]
                out_dict = {
                    k: v[bi] for k, v in outputs.items() if k != "pred_scores_domain"
                }  # one dict for each image
                gt_boxes = torch.tensor(in_dict["org_boxes"])
                gt_pids = torch.tensor(in_dict["ids"], dtype=torch.long)
                det_boxes = out_dict["pred_boxes"].cpu()
                det_boxes_full = out_dict["pred_boxes"].cpu()
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
                    s_recall = tfmat.sum().item() / num_gts
                    s_precision = tfmat.sum().item() / num_dets
                    if s_recall < 1 or s_precision < 1:
                        self._vis_domain(
                            in_dict["file_name"],
                            gt_boxes,
                            torch.tensor(in_dict["ids"]),
                            det_boxes_full,
                            scores_domain,
                            opj(self.svis_dirs[iou_t][sthred], "scores"),
                        )

        else:
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
                        self._vis_samp(
                            in_dict["file_name"],
                            gt_boxes,
                            torch.tensor(in_dict["ids"]),
                            out_dict["pred_boxes"].cpu(),
                            out_dict["pred_scores"].cpu(),
                            iou_t,
                            sthred,
                        )
                    continue

                """ious = torch.zeros((num_gts, num_dets), dtype=torch.float32)
                for i in range(num_gts):
                    for j in range(num_dets):
                        ious[i, j] = get_iou(gt_boxes[i], det_boxes[j])"""
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
                    s_recall = tfmat.sum().item() / num_gts
                    s_precision = tfmat.sum().item() / num_dets
                    if s_recall < 1 or s_precision < 1:
                        self._vis_samp(
                            in_dict["file_name"],
                            gt_boxes,
                            torch.tensor(in_dict["ids"]),
                            out_dict["pred_boxes"].cpu(),
                            out_dict["pred_scores"].cpu(),
                            iou_t,
                            sthred,
                        )

    def evaluate(self):
        if self._distributed:
            if comm.get_local_rank() == 0:
                self.granks_of_local_0s.append(comm.get_rank())
            comm.synchronize()
            save_ranks = comm.all_gather(self.granks_of_local_0s)
            save_ranks = list(set(itertools.chain(*save_ranks)))
            save_rts = {}
            save_gts = {}
            save_gtfs = {}
            for rk in save_ranks:
                g_gts = comm.gather(self.gallery_gts, dst=rk)
                inf_rts = comm.gather(self.inf_results, dst=rk)
                f_gts = comm.gather(self.gallery_fnames, dst=rk)
                if len(inf_rts) > 0:
                    for rts in inf_rts:
                        save_rts.update(rts)
                if len(g_gts) > 0:
                    for gts in g_gts:
                        save_gts.update(gts)
                if len(f_gts) > 0:
                    for fs in f_gts:
                        save_gtfs.update(fs)
            det_scores_per_domain = comm.gather(self.det_scores_per_domain, dst=0)
            det_scores_per_domain = list(itertools.chain(*det_scores_per_domain))
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
            if len(save_rts) == 0 or len(save_gts) == 0 or len(save_gtfs) == 0:
                comm.synchronize()
                return {}

        else:
            save_rts = self.inf_results
            save_gts = self.gallery_gts
            save_gtfs = self.gallery_fnames
            y_true = self.y_trues
            y_score = self.y_scores
            count_gts = self.count_gt
            count_tps = self.count_tp
            count_gts_lb = self.count_gt_lb
            count_tps_lb = self.count_tp_lb
            det_scores_per_domain = self.det_scores_per_domain
        """save_gts = {k: v.to(self._cpu_device) for k, v in save_gts.items()}
        save_rts = {k: v.to(self._cpu_device) for k, v in save_rts.items()}"""
        save_dict = {"gts": save_gts, "infs": save_rts, "gt_fnames": save_gtfs}
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
        if len(det_scores_per_domain) > 0:
            num_domains = len(det_scores_per_domain[0])
            domain_scores = []
            for di in range(num_domains):
                domain_scores.append(
                    torch.cat(
                        [vals[di] for vals in det_scores_per_domain], dim=0
                    ).numpy()
                )
            hist_rgb = domain_hist_img(domain_scores, 10)  # rgb hwc 255
            img_name = str(time.time()) + ".png"
            scrs_dir = opj(self._output_dir, self.dataset_name, "vis", "det")
            if not os.path.exists(scrs_dir):
                os.makedirs(scrs_dir)
            Image.fromarray(hist_rgb).save(opj(scrs_dir, img_name))
        return copy.deepcopy(det_result)

    def _vis_samp(
        self, org_path, tgt_boxes, tgt_ids, all_det_boxes, all_det_scores, iou_t, sthred
    ):
        COLORS = ["r", "g", "b", "y", "c", "m"]
        T_COLORS_BG = {
            "r": "white",
            "g": "white",
            "b": "white",
            "y": "black",
            "c": "black",
            "m": "white",
        }

        from psd2.utils.visualizer import VisImage

        if all_det_boxes.shape[0] > self.topk:
            topk_scores, topk_idxs = torch.topk(all_det_scores[:, 0], self.topk)
            topk_det_boxes = all_det_boxes[topk_idxs]
            det_scores = topk_scores[topk_scores >= sthred]
            det_boxes = topk_det_boxes[topk_scores >= sthred]
        else:
            det_scores_mask = all_det_scores[:, 0] >= sthred
            det_boxes = all_det_boxes[det_scores_mask]
            det_scores = all_det_scores[all_det_scores >= sthred]

        fname = os.path.split(org_path)[-1]
        img = Image.open(org_path)
        vis_org = Visualizer(img.copy())
        vis_det = Visualizer(img.copy())
        det_boxes = det_boxes.cpu().numpy()
        det_scores = det_scores.cpu().numpy().tolist()
        tgt_boxes = tgt_boxes.cpu().numpy()
        tgt_ids = tgt_ids.cpu().numpy().tolist()
        for bi, box in enumerate(tgt_boxes):
            vis_org.draw_box(box)
            id_pos = box[:2]
            vis_org.draw_text(
                str(tgt_ids[bi]), id_pos, horizontal_alignment="left", color="w"
            )
        for i, score in enumerate(det_scores):
            b_clr = COLORS[i % len(COLORS)]
            t_clr = T_COLORS_BG[b_clr]
            vis_det.draw_box(det_boxes[i], edge_color=b_clr)
            vis_det.draw_text(
                "%.2f" % score,
                (det_boxes[i][2], det_boxes[i][1]),
                horizontal_alignment="right",
                color=t_clr,
                bg_color=b_clr,
            )
        ious = box_iou(torch.tensor(tgt_boxes), all_det_boxes)
        # select vis box
        select_scores = torch.zeros_like(ious)
        select_scores[ious > 0.5] = 1.0
        tgt_idxs = torch.arange(0, ious.shape[0], device=ious.device)
        max_iou_idxs = torch.argmax(ious, dim=1)
        select_scores[(tgt_idxs, max_iou_idxs)] = 1.0
        select_scores *= all_det_scores[:, 0][None]
        best_matches = torch.argmax(select_scores, dim=1)
        vis_all = Visualizer(img.copy())
        for bi, box_idx in enumerate(best_matches):
            box = all_det_boxes[box_idx].numpy()
            vis_all.draw_box(box)
            id_pos = box[:2]
            vis_all.draw_text(
                str(tgt_ids[bi]), id_pos, horizontal_alignment="left", color="w"
            )
            score = all_det_scores[box_idx].item()
            vis_all.draw_text(
                "%.2f" % score,
                (box[2], box[1]),
                horizontal_alignment="right",
                color="w",
            )
        img_org = vis_org.get_output().get_image()
        img_det = vis_det.get_output().get_image()
        img_all = vis_all.get_output().get_image()
        img_vis = np.concatenate([img_org, img_all, img_det], axis=1)
        VisImage(img_vis).save(opj(self.svis_dirs[iou_t][sthred], fname))

    def _vis_domain(
        self,
        org_path,
        tgt_boxes,
        tgt_ids,
        det_boxes,
        scores_domain,
        save_dir,
        sthred_low=0.01,
        sthred_mid=0.2,
    ):
        COLORS = ["r", "g", "b", "y", "c", "m"]
        T_COLORS_BG = {
            "r": "white",
            "g": "white",
            "b": "white",
            "y": "black",
            "c": "black",
            "m": "white",
        }

        from psd2.utils.visualizer import VisImage
        import operator

        num_domains = len(scores_domain)
        domain_idxs = list(
            itertools.accumulate([len(dvals) for dvals in scores_domain], operator.add)
        )
        fname = os.path.split(org_path)[-1]
        img = Image.open(org_path)
        vis_org = Visualizer(img.copy())
        vis_det_low = [Visualizer(img.copy()) for _ in range(num_domains)]
        vis_det_high = [Visualizer(img.copy()) for _ in range(num_domains)]
        det_boxes = det_boxes.cpu().numpy()
        det_boxes = np.split(det_boxes, domain_idxs, axis=0)
        tgt_boxes = tgt_boxes.cpu().numpy()
        tgt_ids = tgt_ids.cpu().numpy().tolist()
        for bi, box in enumerate(tgt_boxes):
            vis_org.draw_box(box)
            id_pos = box[:2]
            vis_org.draw_text(
                str(tgt_ids[bi]), id_pos, horizontal_alignment="left", color="w"
            )
        img_org = vis_org.get_output().get_image()
        for di in range(num_domains):
            for i, score in enumerate(scores_domain[di]):
                b_clr = COLORS[i % len(COLORS)]
                t_clr = T_COLORS_BG[b_clr]
                if score >= sthred_low and score < sthred_mid:
                    vis_det_low[di].draw_box(det_boxes[di][i], edge_color=b_clr)
                    vis_det_low[di].draw_text(
                        "%.2f" % score,
                        det_boxes[di][i][:2],
                        horizontal_alignment="left",
                        color=t_clr,
                        bg_color=b_clr,
                    )
                elif score >= sthred_mid:
                    vis_det_high[di].draw_box(det_boxes[di][i], edge_color=b_clr)
                    vis_det_high[di].draw_text(
                        "%.2f" % score,
                        det_boxes[di][i][:2],
                        horizontal_alignment="left",
                        color=t_clr,
                        bg_color=b_clr,
                    )
            img_det_low = vis_det_low[di].get_output().get_image()
            img_det_high = vis_det_high[di].get_output().get_image()
            img_vis = np.concatenate([img_det_low, img_org, img_det_high], axis=1)
            dir_save = opj(save_dir, "d{}".format(di))
            if not os.path.exists(dir_save):
                os.makedirs(dir_save)
            VisImage(img_vis).save(opj(dir_save, fname))


def _trivial_vis(*args, **kw):
    pass
