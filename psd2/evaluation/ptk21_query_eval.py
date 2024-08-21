from .query_evaluator import QueryEvaluator
import copy
import torch
import logging
from torchvision.ops.boxes import box_iou
import numpy as np
from sklearn.metrics import average_precision_score
import psd2.utils.comm as comm
import itertools

logger = logging.getLogger(__name__)


class Ptk21QueryEvaluator(QueryEvaluator):
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
        for bi, in_dict in enumerate(inputs):
            query_dict = in_dict["query"]
            q_imgid = query_dict["image_id"]
            q_pid = query_dict["annotations"][0]["person_id"]
            q_box = query_dict["annotations"][0]["bbox"]
            q_box = torch.tensor(q_box)
            y_trues = {dst: [] for dst in self.det_score_thresh}
            y_scores = {dst: [] for dst in self.det_score_thresh}
            count_gts = {dst: 0 for dst in self.det_score_thresh}
            count_tps = {dst: 0 for dst in self.det_score_thresh}
            # Find all occurence of this query
            # Construct gallery set for this query
            gallery_imgs = []
            query_gts = {}
            for gt_img_id, gt_img_label in self.gts.items():
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

            for dst in self.det_score_thresh:
                if "feat" in query_dict:
                    feat_q = query_dict["feat"].to(self._cpu_device)
                else:
                    query_img_boxes, query_img_feats = self._get_gallery_dets(
                        q_imgid, dst
                    )[:, :4], self._get_gallery_feats(q_imgid, dst)
                    if query_img_boxes.shape[0] == 0:
                        # no detection in this query

                        continue
                        """
                            for item in gallery_imgs:
                                gallery_imname = item["image_id"]
                                g_cid = _get_img_cid(gallery_imname)
                                # some contain the query (gt not empty), some not
                                count_gts[dst] += gallery_imname in query_gts
                                if g_cid != q_cid:
                                    # some contain the query (gt not empty), some not
                                    count_gts_mlv[dst] += gallery_imname in query_gts
                                # compute distance between query and gallery dets
                                if gallery_imname not in self.infs:
                                    continue
                                det, feat_g = self._get_gallery_dets(
                                    gallery_imname, dst
                                ), self._get_gallery_feats(gallery_imname, dst)
                                # no detection in this gallery, skip it
                                if det.shape[0] == 0:
                                    continue
                                label = torch.zeros(feat_g.shape[0], dtype=torch.int)
                                sim = torch.zeros(feat_g.shape[0], dtype=torch.float)
                                y_trues[dst].extend(label.tolist())
                                y_scores[dst].extend(sim.tolist())
                                # multi view
                                if g_cid != q_cid:
                                    y_trues_mlv[dst].extend(label.tolist())
                                    y_scores_mlv[dst].extend(sim.tolist())
                            continue
                            """

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
                    # some contain the query (gt not empty), some not
                    count_gts[dst] += gallery_imname in query_gts
                    # compute distance between query and gallery dets
                    if gallery_imname not in self.infs:
                        continue
                    det, feat_g = self._get_gallery_dets(
                        gallery_imname, dst
                    ), self._get_gallery_feats(gallery_imname, dst)
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
                self.aps[dst].append(ap)
                inds = np.argsort(y_score)[::-1]
                y_score = y_score[inds]
                y_true = y_true[inds]
                self.accs[dst].append([min(1, sum(y_true[:k])) for k in self.topks])

                # 3. Save vis
                self._vis_search(
                    q_imgid,
                    q_box,
                    q_pid,
                    g_img_ids,
                    name2sim,
                    dst,
                    list(query_gts.keys()),
                )
