#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle as pkl
import sys
import torch

"""
Usage:
  # download one of the ResNet{18,34,50,101,152} models from torchvision:
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O r50.pth
  # run the conversion
  ./convert-torchvision-to-d2.py r50.pth r50.pkl
  # Then, use r50.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/r50.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
INPUT:
  FORMAT: "RGB"
  These models typically produce slightly worse results than the
  pre-trained ResNets we use in official configs, which are the
  original ResNet models released by MSRA.
"""

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    if "model" in obj:
        obj = obj["model"]

    newmodel = {}
    for k in list(obj.keys()):
        if k.startswith("backbone"):
            old_k = k
            k = k[9:]
            if "layer" not in k and "down_channel" not in k and "map_channel" not in k:
                k = "stem." + k
            for t in [1, 2, 3, 4]:
                if "r34" in k:
                    k = k.replace("r34_layer{}".format(t), "res{}".format(t + 1))
                elif "os_layer" in k:
                    k = k.replace("os_layer{}".format(t), "conv{}".format(t + 1))
                elif "r50" in k:
                    k = k.replace("r50_layer{}".format(t), "res{}".format(t + 1))
                else:
                    k = k.replace("layer{}".format(t), "res{}".format(t + 1))
            for t in [1, 2, 3]:
                if "os_layer" not in k:
                    k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
            if "os_layer" not in k:
                k = k.replace("downsample.0", "shortcut")
                k = k.replace("downsample.1", "shortcut.norm")
            print(old_k, "->", k)
            newmodel[k] = obj.pop(old_k).detach().numpy()
        elif k.startswith("heads"):
            old_k = k
            if "bottleneck" in k:
                k = k[6:]
                k = k.replace("bottleneck", "bn_neck")
                k = k.replace("bn_neck.0", "bn_neck")
            print(old_k, "->", k)
            newmodel[k] = obj.pop(old_k).detach().numpy()
        else:
            print(k, "->", k)
            newmodel[k] = obj.pop(k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
