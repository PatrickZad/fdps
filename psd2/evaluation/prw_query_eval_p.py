from .query_evaluator import QueryEvaluator
from copy import copy

import psd2.utils.comm as comm

import torch
import logging

import itertools

import numpy as np
from torchvision.ops.boxes import box_iou
from sklearn.metrics import average_precision_score
import copy

import re
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


def _get_img_cid(img_id):
    cn = re.match(r"c(\d+)s.*", img_id)
    cn = int(cn.groups()[0])
    return cn


class PrwQueryEvaluatorP(QueryEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        s_threds=[0.05, 0.2, 0.5, 0.7],
        vis=False,
        hist_only=False,
    ):
        super().__init__(
            dataset_name, distributed, output_dir, s_threds, vis, hist_only
        )
        self.aps_mlv = {st: [] for st in self.det_score_thresh}
        self.accs_mlv = {st: [] for st in self.det_score_thresh}
        self.inf_results = []

    def reset(self):
        super().reset()
        self.aps_mlv = {st: [] for st in self.det_score_thresh}
        self.accs_mlv = {st: [] for st in self.det_score_thresh}

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
                            ...(other optional items)
                        },
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
        for b_rst in eval_worker:
            rst_aps, rst_accs, rst_aps_mlv, rst_accs_mlv = b_rst[0]
            for st in self.det_score_thresh:
                self.aps[st].extend(rst_aps[st])
                self.accs[st].extend(rst_accs[st])
                self.aps_mlv[st].extend(rst_aps_mlv[st])
                self.accs_mlv[st].extend(rst_accs_mlv[st])
        mix_eval_results = super().evaluate()
        if self._distributed:
            comm.synchronize()
            aps_all = comm.gather(self.aps_mlv, dst=0)
            accs_all = comm.gather(self.accs_mlv, dst=0)
            if not comm.is_main_process():
                return {}
            aps = {}
            accs = {}
            for dst in self.det_score_thresh:
                aps[dst] = list(itertools.chain(*[ap[dst] for ap in aps_all]))
                accs[dst] = list(itertools.chain(*[acc[dst] for acc in accs_all]))
        else:
            aps = self.aps_mlv
            accs = self.accs_mlv

        for dst in self.det_score_thresh:
            logger.info(
                "Multi-view Search eval_{:.2f} on {} queries. ".format(
                    dst, len(aps[dst])
                )
            )
            mAP = np.mean(aps[dst])
            mix_eval_results["search"].update({"mAP_{:.2f}_mlv".format(dst): mAP})
            acc = np.mean(accs[dst], axis=0)
            # logger.info(str(acc))
            for i, v in enumerate(acc.tolist()):
                # logger.info("{:.2f} on {} acc. ".format(v, i))
                k = self.topks[i]
                mix_eval_results["search"].update(
                    {"top{:2d}_{:.2f}_mlv".format(k, dst): v}
                )
        return copy.deepcopy(mix_eval_results)


class EvaluatorDataset(Dataset):
    def __init__(self, eval_inputs, eval_ref):
        self.eval_inputs = eval_inputs
        self.eval_ref = eval_ref

    def __len__(self):
        return len(self.eval_inputs)

    def __getitem__(self, idx):
        inputs = self.eval_inputs[idx]
        rst_aps_mlv = {st: [] for st in self.eval_ref.det_score_thresh}
        rst_accs_mlv = {st: [] for st in self.eval_ref.det_score_thresh}
        rst_aps = {st: [] for st in self.eval_ref.det_score_thresh}
        rst_accs = {st: [] for st in self.eval_ref.det_score_thresh}
        for bi, in_dict in enumerate(inputs):
            query_dict = in_dict["query"]
            q_imgid = query_dict["image_id"]
            q_pid = query_dict["annotations"][0]["person_id"]
            q_box = query_dict["annotations"][0]["bbox"]
            q_box = torch.tensor(q_box)
            q_cid = _get_img_cid(q_imgid)
            y_trues = {dst: [] for dst in self.eval_ref.det_score_thresh}
            y_scores = {dst: [] for dst in self.eval_ref.det_score_thresh}
            count_gts = {dst: 0 for dst in self.eval_ref.det_score_thresh}
            count_tps = {dst: 0 for dst in self.eval_ref.det_score_thresh}
            y_trues_mlv = {dst: [] for dst in self.eval_ref.det_score_thresh}
            y_scores_mlv = {dst: [] for dst in self.eval_ref.det_score_thresh}
            count_gts_mlv = {dst: 0 for dst in self.eval_ref.det_score_thresh}
            count_tps_mlv = {dst: 0 for dst in self.eval_ref.det_score_thresh}
            # Find all occurence of this query
            # Construct gallery set for this query
            gallery_imgs = []
            query_gts = {}
            for gt_img_id, gt_img_label in self.eval_ref.gts.items():
                if gt_img_id != q_imgid:
                    gt_boxes = gt_img_label[:, :4]
                    gt_ids = gt_img_label[:, 4].long()
                    gallery_imgs.append(
                        {
                            "image_id": gt_img_id,
                            "boxes": gt_boxes,
                            "ids": gt_ids,
                        }
                    )
                    if q_pid in gt_ids.tolist():
                        query_gts[gt_img_id] = gt_boxes[gt_ids == q_pid].squeeze(0)

            for dst in self.eval_ref.det_score_thresh:
                if "feat" in query_dict:
                    feat_q = query_dict["feat"]
                else:
                    query_img_boxes, query_img_feats = self.eval_ref._get_gallery_dets(
                        q_imgid, dst
                    )[:, :4], self.eval_ref._get_gallery_feats(q_imgid, dst)
                    if query_img_boxes.shape[0] == 0:
                        # no detection in this query image
                        continue
                    ious = box_iou(q_box[None, :], query_img_boxes)
                    max_iou, nmax = torch.max(ious, dim=1)
                    feat_q = query_img_feats[nmax.item()]

                # feat_q = F.normalize(feat_q[None]).squeeze(0)  # NOTE keep post norm
                name2sim = {}
                # save for vis
                g_img_ids = []
                # 1. Go through the gallery samples defined by the protocol
                for item in gallery_imgs:
                    gallery_imname = item["image_id"]
                    g_cid = _get_img_cid(gallery_imname)
                    # some contain the query (gt not empty), some not
                    count_gts[dst] += gallery_imname in query_gts
                    if g_cid != q_cid:
                        # some contain the query (gt not empty), some not
                        count_gts_mlv[dst] += gallery_imname in query_gts
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
                    # save for vis
                    g_img_ids.append(gallery_imname)

                    label = torch.zeros(sim.shape[0], dtype=torch.int)
                    if gallery_imname in query_gts:
                        gt = query_gts[gallery_imname]
                        w, h = gt[2] - gt[0], gt[3] - gt[1]
                        iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                        inds = torch.argsort(sim)
                        inds = inds.tolist()[::-1]
                        inds = torch.tensor(inds, dtype=torch.long)
                        sim = sim[inds]
                        det = det[inds]
                        # only set the first matched det as true positive
                        for j, roi in enumerate(det[:, :4]):
                            if (
                                box_iou(roi[None, :], gt[None, :]).squeeze().item()
                                >= iou_thresh
                            ):
                                label[j] = 1
                                count_tps[dst] += 1
                                if q_cid != g_cid:
                                    count_tps_mlv[dst] += 1
                                break
                    y_trues[dst].extend(label.tolist())
                    y_scores[dst].extend(sim.tolist())
                    # multi view
                    if g_cid != q_cid:
                        y_trues_mlv[dst].extend(label.tolist())
                        y_scores_mlv[dst].extend(sim.tolist())

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
                # mlv
                y_score_mlv = np.asarray(y_scores_mlv[dst])
                y_true_mlv = np.asarray(y_trues_mlv[dst])
                assert count_tps_mlv[dst] <= count_gts_mlv[dst]
                recall_rate_mlv = count_tps_mlv[dst] * 1.0 / count_gts_mlv[dst]
                ap_mlv = (
                    0
                    if count_tps_mlv[dst] == 0
                    else average_precision_score(y_true_mlv, y_score_mlv)
                    * recall_rate_mlv
                )
                rst_aps_mlv[dst].append(ap_mlv)
                inds_mlv = np.argsort(y_score_mlv)[::-1]
                y_score_mlv = y_score_mlv[inds_mlv]
                y_true_mlv = y_true_mlv[inds_mlv]
                rst_accs_mlv[dst].append(
                    [min(1, sum(y_true_mlv[:k])) for k in self.eval_ref.topks]
                )
                # 3. Save vis
                self.eval_ref._vis_search(
                    q_imgid,
                    q_box,
                    q_pid,
                    g_img_ids,
                    name2sim,
                    dst,
                    list(query_gts.keys()),
                )
        return rst_aps, rst_accs, rst_aps_mlv, rst_accs_mlv
