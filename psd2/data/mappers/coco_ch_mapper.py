import psd2.data.transforms as dT
import torchvision.transforms as tT
import torchvision.transforms.functional as tvtF
from ..detection_utils import read_image
import numpy as np
import torch
import torch.nn as nn


class COCOCHMapper(object):
    def __init__(self, cfg, is_train) -> None:
        self.is_train = is_train
        img_mean = cfg.MODEL.PIXEL_MEAN
        img_std = cfg.MODEL.PIXEL_STD
        if self.is_train:
            self.augs_box = dT.AugmentationList(
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
            self.augs_color = tT.transforms.Compose(
                [
                    tT.RandomApply([tT.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    tT.RandomGrayscale(p=0.2),
                    tT.RandomApply([GaussianBlur([0.1, 2.0])]),
                ]
            )
        else:
            self.augs_box = dT.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                sample_style="choice",
            )
        self.norm = tT.Normalize(mean=img_mean, std=img_std)

        self.to_pil = tT.ToPILImage()
        self.img_fmt = cfg.INPUT.FORMAT

    def __call__(self, img_dict):
        return self._common_map(img_dict)

    def _common_map(self, img_dict):
        # TODO use standared structure
        img_path = img_dict["file_name"]
        img_arr = read_image(img_path, self.img_fmt)
        orgh, orgw = img_arr.shape[:2]
        boxes = []
        ids = []
        for ann in img_dict["annotations"]:
            box = ann["bbox"]
            boxes.append(box)
            ids.append(ann["person_id"])
        org_boxes = np.array(boxes, dtype=np.float32)
        # img 1
        aug_input = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
        transforms = self.augs_box(aug_input)
        aug_img = aug_input.image
        aug_boxes = aug_input.boxes
        aug_h, aug_w = aug_img.shape[0], aug_img.shape[1]
        if self.is_train:
            aug_img = self.augs_color(self.to_pil(aug_img))
            # img 2
            aug_input_2 = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
            transforms_2 = self.augs_box(aug_input_2)
            aug_img_2 = aug_input_2.image
            aug_h_2, aug_w_2 = aug_img_2.shape[0], aug_img_2.shape[1]
            aug_boxes_2 = aug_input_2.boxes
            aug_img_2 = self.augs_color(self.to_pil(aug_img_2))
            imgt_2 = tvtF.to_tensor(aug_img_2.copy())
            img_2_dict = {
                "file_name": img_path,
                "image_id": img_dict["image_id"],
                "image": self.norm(imgt_2),
                "width": aug_w_2,
                "height": aug_h_2,
                "boxes": aug_boxes_2,
                "ids": ids,
                "org_width": img_arr.shape[1],
                "org_height": img_arr.shape[0],
                "org_boxes": org_boxes,
            }
            imgt_1 = tvtF.to_tensor(aug_img.copy())
            return {
                "file_name": img_path,
                "image_id": img_dict["image_id"],
                "image": self.norm(imgt_1),
                "width": aug_w,
                "height": aug_h,
                "boxes": aug_boxes,
                "ids": ids,
                "org_width": img_arr.shape[1],
                "org_height": img_arr.shape[0],
                "org_boxes": org_boxes,
                "img2_input": img_2_dict,
            }
        else:
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


class COCOCH2vMapper(object):
    def __init__(self, cfg, is_train) -> None:
        self.is_train = is_train
        img_mean = cfg.MODEL.PIXEL_MEAN
        img_std = cfg.MODEL.PIXEL_STD
        if self.is_train:
            self.augs_box1 = dT.AugmentationList(
                [
                    dT.ResizeShortestEdge(
                        cfg.INPUT.MIN_SIZE_TRAIN,
                        cfg.INPUT.MAX_SIZE_TRAIN,
                        size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                        sample_style="choice",
                    ),
                ]
            )
            self.augs_box2 = dT.AugmentationList(
                [
                    dT.ResizeShortestEdge(
                        cfg.INPUT.MIN_SIZE_TRAIN,
                        cfg.INPUT.MAX_SIZE_TRAIN,
                        size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                        sample_style="choice",
                    ),
                    dT.RandomFlip(prob=1.0, horizontal=True, vertical=False),
                ]
            )
            self.augs_color1 = tT.transforms.Compose(
                [
                    tT.RandomApply([tT.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    tT.RandomGrayscale(p=0.2),
                    tT.RandomApply([GaussianBlur([0.1, 2.0])]),
                ]
            )
            self.augs_color2 = self.augs_color1
        else:
            self.augs_box1 = dT.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                sample_style="choice",
            )
        self.norm = tT.Normalize(mean=img_mean, std=img_std)

        self.to_pil = tT.ToPILImage()
        self.img_fmt = cfg.INPUT.FORMAT

    def __call__(self, img_dict):
        return self._common_map(img_dict)

    def _common_map(self, img_dict):
        # TODO use standared structure
        img_path = img_dict["file_name"]
        img_arr = read_image(img_path, self.img_fmt)
        orgh, orgw = img_arr.shape[:2]
        boxes = []
        ids = []
        for ann in img_dict["annotations"]:
            box = ann["bbox"]
            boxes.append(box)
            ids.append(ann["person_id"])
        org_boxes = np.array(boxes, dtype=np.float32)
        # img 1
        aug_input = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
        transforms = self.augs_box1(aug_input)
        aug_img = aug_input.image
        aug_boxes = aug_input.boxes
        aug_h, aug_w = aug_img.shape[0], aug_img.shape[1]
        if self.is_train:
            aug_img = self.augs_color1(self.to_pil(aug_img))
            # img 2
            aug_input_2 = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
            transforms_2 = self.augs_box2(aug_input_2)
            aug_img_2 = aug_input_2.image
            aug_h_2, aug_w_2 = aug_img_2.shape[0], aug_img_2.shape[1]
            aug_boxes_2 = aug_input_2.boxes
            aug_img_2 = self.augs_color2(self.to_pil(aug_img_2))
            imgt_2 = tvtF.to_tensor(aug_img_2.copy())
            if imgt_2.shape[0] == 1:
                print("get")
            img_2_dict = {
                "file_name": img_path,
                "image_id": img_dict["image_id"],
                "image": self.norm(imgt_2),
                "width": aug_w_2,
                "height": aug_h_2,
                "boxes": aug_boxes_2,
                "ids": ids,
                "org_width": img_arr.shape[1],
                "org_height": img_arr.shape[0],
                "org_boxes": org_boxes,
            }
            imgt_1 = tvtF.to_tensor(aug_img.copy())
            if imgt_1.shape[0] == 1:
                print("get")
            return {
                "file_name": img_path,
                "image_id": img_dict["image_id"],
                "image": self.norm(imgt_1),
                "width": aug_w,
                "height": aug_h,
                "boxes": aug_boxes,
                "ids": ids,
                "org_width": img_arr.shape[1],
                "org_height": img_arr.shape[0],
                "org_boxes": org_boxes,
                "img2_input": img_2_dict,
            }
        else:
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


# from moco
from PIL import ImageFilter, ImageOps
import random


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
