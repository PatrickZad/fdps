import logging
from os.path import join as opj
import numpy as np
from psd2.structures import BoxMode
from scipy.io import loadmat
import os
from tqdm import tqdm
import logging
import json
import copy
import re

"""
CDPS: train frames: 10366, ids: 349 test frames: 10049, test_ids: 259 , query_ids: 150 TODO duplicate
"""

logger = logging.getLogger(__name__)  # setup_logger()

__all__ = ["load_cdps", "subset_names"]
subset_names = ("Train", "Gallery", "Query")
id_map = {}
next_pid = 0


def load_cdps(dataset_dir, subset="Train"):
    """
    Returns:
        train/test set meta dict
            [
                {
                    "file_name": image paths,
                    "image_id": image name,
                    "annotations":
                        [
                            {
                                "bbox": person xyxy_rel boxe,
                                "bbox_mode": format of bbox
                                "person_id":  person id
                            },...
                        ]
                },
                ...
            ]
        or
        query meta dict
            [
                {
                    "query":
                        {
                            "file_name": image paths,
                            "image_id": image name,
                            "annotations":
                                [
                                    {
                                        "bbox": person xyxy_rel boxes,
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
                                            "bbox": person xyxy_rel boxes,
                                            "bbox_mode": format of bbox,                                                "person_id":  person id
                                        }
                                    ],
                            },
                            ...
                        ]
                }
            ]
    """
    assert subset in subset_names
    if subset == subset_names[0]:
        return _build_sub(dataset_dir, "train")
    if subset == subset_names[1]:
        return _build_sub(dataset_dir, "test")
    if subset == subset_names[2]:
        # build gallery
        test_annos = _build_sub(dataset_dir, "test", as_dicts=True)
        qg_dicts = []
        for s_dir in os.listdir(opj(dataset_dir, "test")):
            for g_dir in os.listdir(opj(dataset_dir, "test", s_dir)):
                g_path = opj(dataset_dir, "test", s_dir, g_dir)
                with open(opj(g_path, "search-new.json"), "r") as jf:
                    qg_str = jf.read()
                qg_str_dicts = json.loads(qg_str)
                for idqg_dict in qg_str_dicts:
                    pid_str = list(idqg_dict.keys())[0]
                    qg_dict = idqg_dict[pid_str]
                    qimg_id = qg_dict["query"]
                    pid = id_map[int(pid_str)]
                    q_box = _get_box(test_annos, qimg_id, pid)
                    q_dict = {
                        "file_name": test_annos[qimg_id]["file_name"],
                        "image_id": qimg_id,
                        "annotations": [
                            {
                                "bbox": q_box,
                                "bbox_mode": BoxMode.XYXY_REL,
                                "person_id": pid,
                            }
                        ],
                    }
                    g_dicts = []
                    for gimg_id in qg_dict["gallery"]:
                        gbox = _get_box(test_annos, gimg_id, pid)
                        if gbox.size < 4:
                            gpid = None
                        else:
                            gpid = pid
                        g_dicts.append(
                            {
                                "file_name": test_annos[gimg_id]["file_name"],
                                "image_id": gimg_id,
                                "annotations": [
                                    {
                                        "bbox": gbox,
                                        "bbox_mode": BoxMode.XYXY_REL,
                                        "person_id": gpid,
                                    }
                                ],
                            }
                        )
                    qg_dicts.append({"query": q_dict, "gallery": g_dicts})
        return qg_dicts


def _get_box(all_annos_dicts, img_id, pid):
    annos = all_annos_dicts[img_id]
    for anno in annos["annotations"]:
        box = anno["bbox"]
        box_pid = anno["person_id"]
        if pid == box_pid:
            """box[:2] = box[:2] - box[2:] / 2
            box[2:] = box[:2] + box[2:]  # xyxy_rel"""
            return box
    return np.array([], dtype=np.float32)


def _build_sub(dataset_dir, subset, as_dicts=False):
    subname = subset.lower()
    # build train/test
    if as_dicts:
        img_dicts = {}
    else:
        img_dicts = []
    with tqdm(total=64) as pbar:
        logger.info("Loading CDPS c annotations:")
        for s_dir in os.listdir(opj(dataset_dir, subname)):
            for g_dir in os.listdir(opj(dataset_dir, subname, s_dir)):
                g_path = opj(dataset_dir, subname, s_dir, g_dir)
                if as_dicts:
                    for fn in os.listdir(g_path):
                        if re.match(r"c\d.json", fn):
                            img_dicts.update(
                                _anno_format(
                                    opj(dataset_dir, subname),
                                    opj(g_path, fn),
                                    as_dicts=True,
                                )
                            )
                            pbar.update(1)
                else:
                    for fn in os.listdir(g_path):
                        if re.match(r"c\d.json", fn):
                            img_dicts.extend(
                                _anno_format(opj(dataset_dir, subname), opj(g_path, fn))
                            )
                            pbar.update(1)
    return img_dicts


def _anno_format(dataset_dir, js_path, as_dicts=False):
    global id_map
    global next_pid
    with open(js_path, "r") as jf:
        js_str = jf.read()
    anno_dicts = json.loads(js_str)
    if as_dicts:
        img_dicts = {}
    else:
        img_dicts = []
    for js_dict in anno_dicts:
        img_dict = copy.deepcopy(js_dict)
        fn = opj(dataset_dir, img_dict.pop("file_path"))
        ids_str = img_dict.pop("ids")
        bboxes_str = img_dict.pop("bboxes")
        ids = np.array(ids_str, dtype=np.int32)
        ids[ids < 0] = -1
        ids = ids.tolist()
        for i, pid in enumerate(ids):
            if pid > -1:
                if pid in id_map.keys():
                    ids[i] = id_map[pid]
                else:
                    id_map[pid] = next_pid
                    ids[i] = next_pid
                    next_pid += 1
        bboxes = np.array(bboxes_str, dtype=np.float32)  # ccwh_rel
        bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:] / 2
        bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]  # xyxy_rel"""
        img_dict["file_name"] = fn
        img_dict["annotations"] = [
            {"bbox": bboxes[i], "bbox_mode": BoxMode.XYXY_REL, "person_id": ids[i]}
            for i in range(bboxes.shape[0])
        ]
        if as_dicts:
            img_dicts[img_dict["image_id"]] = img_dict
        else:
            img_dicts.append(img_dict)
    return img_dicts
