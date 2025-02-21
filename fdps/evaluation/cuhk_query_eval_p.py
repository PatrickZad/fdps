from copy import copy
import enum
from math import inf
from typing import Dict, OrderedDict
import fdps.utils.comm as comm

import torch
import logging

import numpy as np
from torchvision.ops.boxes import box_iou
from sklearn.metrics import average_precision_score

import logging
from .query_evaluator import QueryEvaluator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)  # setup_logger()


class CuhkQueryEvaluatorP(QueryEvaluator):
    def __init__(self, *args, **kws) -> None:
        super().__init__(*args, **kws)
        self.inf_results = []

    def process(self, inputs, outputs):
        """
        inputs:
            [
                {
                    "query":
                        {
                            "file_name": image paths,
                            "image_id": image name,
                            "annotations":
                                [
                                    {
                                        "bbox": person xyxy_abs box,
                                        "bbox_mode": format of bbox
                                        "person_id":  person id
                                    }
                                ],
                        },
                    "gallery":
                            [
                                {
                                    "file_name": image paths,
                                    "image_id": image name,
                                    "annotations":
                                        [
                                            {
                                                "bbox": person xyxy_abs boxes,
                                                "bbox_mode": format of bbox
                                                "person_id":  person id
                                            }
                                        ],
                                },
                                ...
                            ]
                },
                ...
            ]
        outputs:
            dummy inputs
        """
        for item in inputs:
            q_dict = item["query"]
            q_dict.pop("image")
            if "feat" in q_dict:
                q_dict["feat"] = q_dict["feat"].to(self._cpu_device)
        self.inf_results.append(inputs)

    def evaluate(self):
        eval_dataset = EvaluatorDataset(self.inf_results, self)
        eval_worker = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=lambda x: x,
        )
        print("Parallel evaluating")
        with tqdm(total=len(eval_worker)) as pbar:
            for b_rst in eval_worker:
                rst_aps, rst_accs = b_rst[0]
                for st in self.det_score_thresh:
                    self.aps[st].extend(rst_aps[st])
                    self.accs[st].extend(rst_accs[st])
                pbar.update(1)
        return super().evaluate()


class EvaluatorDataset(Dataset):
    def __init__(self, eval_inputs, eval_wref):
        self.eval_inputs = eval_inputs
        self.eval_ref = eval_wref

    def __len__(self):
        return len(self.eval_inputs)

    def __getitem__(self, idx):
        inputs = self.eval_inputs[idx]
        rst_aps = {st: [] for st in self.eval_ref.det_score_thresh}
        rst_accs = {st: [] for st in self.eval_ref.det_score_thresh}
        for bi, in_dict in enumerate(inputs):
            gallery_dicts = in_dict["gallery"]
            query_dict = in_dict["query"]
            q_imgid = query_dict["image_id"]
            q_pid = query_dict["annotations"][0]["person_id"]
            q_box = query_dict["annotations"][0]["bbox"]
            q_box = torch.tensor(q_box)
            y_trues = {dst: [] for dst in self.eval_ref.det_score_thresh}
            y_scores = {dst: [] for dst in self.eval_ref.det_score_thresh}
            count_gts = {dst: 0 for dst in self.eval_ref.det_score_thresh}
            count_tps = {dst: 0 for dst in self.eval_ref.det_score_thresh}
            for dst in self.eval_ref.det_score_thresh:
                if "feat" in query_dict:
                    feat_q = query_dict["feat"]
                else:
                    query_img_boxes, query_img_feats = self.eval_ref._get_gallery_dets(
                        q_imgid, dst
                    )[:, :4], self.eval_ref._get_gallery_feats(q_imgid, dst)
                    if query_img_boxes.shape[0] == 0:
                        # no detection in this query
                        # continue

                        for item in gallery_dicts:
                            gallery_imname = item["image_id"]
                            # some contain the query (gt not empty), some not
                            gt = item["annotations"][0]["bbox"]
                            gt = torch.tensor(gt)
                            count_gts[dst] += gt.shape[0] > 0
                            # compute distance between query and gallery dets
                            if gallery_imname not in self.eval_ref.infs:
                                continue
                            det, feat_g = self.eval_ref._get_gallery_dets(
                                gallery_imname, dst
                            ), self.eval_ref._get_gallery_feats(gallery_imname, dst)
                            # no detection in this gallery, skip it
                            if det.shape[0] == 0:
                                continue
                            label = torch.zeros(feat_g.shape[0], dtype=torch.int)
                            sim = torch.zeros(feat_g.shape[0], dtype=torch.float)
                            y_trues[dst].extend(label.tolist())
                            y_scores[dst].extend(sim.tolist())
                        continue

                    ious = box_iou(q_box[None, :], query_img_boxes)
                    max_iou, nmax = torch.max(ious, dim=1)
                    feat_q = query_img_feats[nmax.item()]

                # feat_q = F.normalize(feat_q[None]).squeeze(0)  # NOTE keep post norm
                name2sim = {}


                # 1. Go through the gallery samples defined by the protocol
                for item in gallery_dicts:
                    gallery_imname = item["image_id"]
                    # some contain the query (gt not empty), some not
                    gt = item["annotations"][0]["bbox"]
                    gt = torch.tensor(gt)
                    count_gts[dst] += gt.shape[0] > 0
                    # compute distance between query and gallery dets
                    if gallery_imname not in self.eval_ref.infs:
                        continue
                    det, feat_g = self.eval_ref._get_gallery_dets(
                        gallery_imname, dst
                    ), self.eval_ref._get_gallery_feats(gallery_imname, dst)
                    # no detection in this gallery, skip it
                    if det.shape[0] == 0:
                        continue
                    # get L2-normalized feature matrix NxD
                    # feat_g = F.normalize(feat_g)  # NOTE keep post norm
                    # compute cosine similarities
                    sim = torch.mm(feat_g, feat_q[:, None]).squeeze(1)  # n x 1 -> n
                    if gallery_imname in name2sim:
                        continue
                    name2sim[gallery_imname] = sim

                    label = torch.zeros(sim.shape[0], dtype=torch.int)
                    if gt.shape[0] > 0:
                        w, h = gt[2] - gt[0], gt[3] - gt[1]
                        iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                        inds = torch.argsort(sim)
                        inds = inds.tolist()[::-1]
                        inds = torch.tensor(inds, dtype=torch.long)
                        sim = name2sim[gallery_imname][inds]
                        det = det[inds]
                        # only set the first matched det as true positive
                        for j, roi in enumerate(det[:, :4]):
                            if (
                                box_iou(roi[None, :], gt[None, :]).squeeze().item()
                                >= iou_thresh
                            ):
                                label[j] = 1
                                count_tps[dst] += 1
                                break
                    y_trues[dst].extend(label.tolist())
                    y_scores[dst].extend(sim.tolist())

                # 2. Compute AP for this probe (need to scale by recall rate)

                y_score = np.asarray(y_scores[dst])
                y_true = np.asarray(y_trues[dst])
                assert count_tps[dst] <= count_gts[dst]
                recall_rate = count_tps[dst] * 1.0 / count_gts[dst]
                ap = (
                    0
                    if count_tps[dst] == 0
                    else average_precision_score(y_true, y_score) * recall_rate
                )
                rst_aps[dst].append(ap)
                inds = np.argsort(y_score)[::-1]
                y_score = y_score[inds]
                y_true = y_true[inds]
                rst_accs[dst].append(
                    [min(1, sum(y_true[:k])) for k in self.eval_ref.topks]
                )

        return rst_aps, rst_accs
