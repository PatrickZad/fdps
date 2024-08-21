import psd2.data.transforms as dT
import torchvision.transforms as tT
from PIL import Image
import numpy as np
import torch

from psd2.structures.boxes import BoxMode
from ..detection_utils import read_image
from copy import deepcopy
import numbers
import warnings
from typing import Tuple, List, Optional
from torch import Tensor
import math


class CdpsMapper(object):
    def __init__(self, cfg, is_train) -> None:
        self.is_train = is_train
        img_mean = cfg.MODEL.PIXEL_MEAN
        img_std = cfg.MODEL.PIXEL_STD
        if self.is_train:
            self.augs = dT.AugmentationList(
                [
                    dT.ResizeShortestEdge(
                        cfg.INPUT.MIN_SIZE_TRAIN,
                        cfg.INPUT.MAX_SIZE_TRAIN,
                        size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                        sample_style="choice",
                    ),
                    dT.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                ]
            )
        else:
            self.augs = dT.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                sample_style="choice",
            )
        self.totensor_norm = tT.Compose(
            [
                tT.ToTensor(),
                tT.Normalize(mean=img_mean, std=img_std),
            ]
        )

    def __call__(self, img_dict):
        if "query" in img_dict.keys():
            return img_dict
        else:
            return self._common_map(img_dict)

    def _common_map(self, img_dict):
        # TODO use standared structure
        img_path = img_dict["file_name"]
        img_arr = read_image(img_path)
        orgh, orgw = img_arr.shape[:2]
        boxes = []
        ids = []
        for ann in img_dict["annotations"]:
            box = ann["bbox"] * np.array([orgw, orgh] * 2)
            boxes.append(box)
            ids.append(ann["person_id"])
        org_boxes = np.array(boxes, dtype=np.float32)
        aug_input = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
        aug_img = aug_input.image
        aug_boxes = aug_input.boxes
        return {
            "file_name": img_path,
            "image_id": img_dict["image_id"],
            "image": self.totensor_norm(aug_img.copy()),
            "width": aug_img.shape[1],
            "height": aug_img.shape[0],
            "boxes": aug_boxes,
            "ids": ids,
            "org_width": img_arr.shape[1],
            "org_height": img_arr.shape[0],
            "org_boxes": org_boxes,
        }


class CdpsSearchMapperInfQuery(CdpsMapper):
    def __call__(self, img_dict):
        """
        For query, return
        {
            "query":
                {
                    "file_name": image paths,
                    "image_id": image name,
                    "image": augmented image tensor,
                    "width","height",
                    "boxes": augmented boxes,
                    "ids","org_width","org_height","org_boxes",
                    "annotations":
                        [
                            {
                                "bbox": person xyxy_abs boxes,
                                "bbox_mode": format of bbox
                                "person_id":  person id
                            }
                        ],
                },
            "gallery": (Optional)
                [
                    {
                        "file_name": image paths,
                        "image_id": image name,
                        "annotations":
                            [
                                {
                                    "bbox": person xyxy_abs boxes,
                                    "bbox_mode": format of bbox,                                                "person_id":  person id
                                }
                            ],
                    },
                    ...
                ]
        }
        """
        # For compatibility
        if "query" in img_dict.keys():
            q_img_dict = img_dict["query"]
            mapped_img_dict = self._common_map(q_img_dict)
            org_dict = deepcopy(img_dict)
            q_dict = org_dict["query"]
            for ann in q_dict["annotations"]:
                imgw, imgh = Image.open(q_img_dict["file_name"]).size
                ann["bbox"] = ann["bbox"] * np.array([imgw, imgh] * 2)
                ann["bbox_mode"] == BoxMode.XYXY_ABS
            mapped_img_dict["annotations"] = q_dict["annotations"]
            org_dict["query"] = mapped_img_dict
            # gallery box
            g_dicts = org_dict["gallery"]
            for ann_dict in g_dicts:
                ann = ann_dict["annotations"][0]
                imgw, imgh = Image.open(ann_dict["file_name"]).size
                rel_box = ann["bbox"]
                if rel_box.shape[0] < 4:
                    continue
                ann["bbox"] = rel_box * np.array([imgw, imgh] * 2)
                ann["bbox_mode"] == BoxMode.XYXY_ABS
            return org_dict
        else:
            return self._common_map(img_dict)


class RandomErasing:
    """Randomly selects a rectangle region in a person and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against bounding box.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.

    Returns:
        Erased Image.

    Example: Used after Normalize
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError(
                "Argument value should be either a number or str or a sequence"
            )
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

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        # cast self.value to script acceptable type
        if isinstance(self.value, (int, float)):
            self.value = [
                self.value,
            ]
        elif isinstance(self.value, tuple):
            self.value = list(self.value)

    @staticmethod
    def get_params(box, channels, scale, ratio, value=None):
        """Get parameters for ``erase`` for a random erasing."""
        box = np.round(box).astype(np.int32)
        img_c, img_h, img_w = channels, box[3] - box[1], box[2] - box[0]
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
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i + box[0], j + box[1], h, w, v

        # Return original image
        return box[0], box[1], 0, 0, 0

    def forward(self, img, boxes):
        vals = np.random.rand(boxes.shape[0])
        for i in range(boxes.shape[0]):
            if vals[i] < self.p:
                x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio)
                return self.erase(img, x, y, h, w, v)
        return img

    def erase(
        img: Tensor, i: int, j: int, h: int, w: int, v: Tensor, inplace=False
    ) -> Tensor:
        """Erase the input Tensor Image with given value.
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
            raise TypeError("img should be Tensor Image. Got {}".format(type(img)))

        if not inplace:
            img = img.clone()

        img[..., i : i + h, j : j + w] = v
        return img
