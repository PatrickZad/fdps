import logging
import os
from os.path import join as opj
import numpy as np
from psd2.structures import BoxMode
import os
from tqdm import tqdm
import logging
import json
import copy

# train imgs 42861, ids, 5381 ulb 0
logger = logging.getLogger(__name__)  # setup_logger()

__all__ = ["load_ptk21", "subset_names"]
subset_names = ("Train", "Gallery", "Query")
anno_dir = "annotations/posetrack_person_search"
id_map_dict = {}
id_count_dict = {}
next_pid = 0


def _get_map_id(org_id):
    global id_map_dict
    global next_pid
    if org_id in id_map_dict:
        new_id = id_map_dict[org_id]
        id_count_dict[new_id] += 1
        return new_id
    else:
        id_map_dict[org_id] = next_pid
        id_count_dict[next_pid] = 1
        next_pid += 1
        return id_map_dict[org_id]


def load_ptk21(dataset_dir, subset="Train"):
    assert subset in subset_names
    logger.info("Loading PoseTrack21 {} :".format(subset))
    if subset == subset_names[0]:
        with open(opj(dataset_dir, anno_dir, "train.json"), "r") as jf:
            ann_org = json.load(jf)
    elif subset == subset_names[1]:
        with open(opj(dataset_dir, anno_dir, "val.json"), "r") as jf:
            ann_org = json.load(jf)
    else:
        with open(opj(dataset_dir, anno_dir, "query.json"), "r") as jf:
            ann_org = json.load(jf)
    imgs = ann_org["images"]
    anns = ann_org["annotations"]
    anno_dicts = {
        img_["id"]: {
            "file_name": opj(dataset_dir, img_["file_name"]),
            "image_id": img_["file_name"],
            "annotations": [],
        }
        for img_ in imgs
    }
    if subset == subset_names[2]:
        ann_list = []
        with tqdm(total=len(anns)) as pbar:
            for ann in anns:
                img_id = ann["image_id"]
                person_id = ann["person_id"]
                xywh_bbox = np.array(ann["bbox"], dtype=np.float32)
                xyxy_bbox = xywh_bbox.copy()
                xyxy_bbox[2:] += xywh_bbox[:2]
                ann_dict = {
                    "bbox": xyxy_bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "person_id": person_id,
                }
                img_ann = copy.deepcopy(anno_dicts[img_id])
                img_ann["annotations"].append(ann_dict)
                ann_list.append({"query": img_ann})
    else:
        with tqdm(total=len(anns)) as pbar:
            for ann in anns:
                img_id = ann["image_id"]
                org_pid = ann["person_id"]
                if subset == subset_names[0]:
                    person_id = _get_map_id(org_pid)
                else:
                    person_id = org_pid
                xywh_bbox = np.array(ann["bbox"], dtype=np.float32)
                xyxy_bbox = xywh_bbox.copy()
                xyxy_bbox[2:] += xywh_bbox[:2]
                ann_dict = {
                    "bbox": xyxy_bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "person_id": person_id,
                }
                anno_dicts[img_id]["annotations"].append(ann_dict)
                pbar.update(1)
        ann_list = list(anno_dicts.values())
    return ann_list
