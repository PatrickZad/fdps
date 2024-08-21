# Copyright (c) Facebook, Inc. and its affiliates.
from .build import build_backbone, BACKBONE_REGISTRY  # noqa F401 isort:skip

from .backbone import Backbone
from .fpn import FPN
from .fpn_mr import MrFPN
from .regnet import RegNet
from .resnet import (
    BasicStem,
    ResNet,
    ResNetBlockBase,
    build_resnet_backbone,
    make_stage,
    BottleneckBlock,
    build_resnet_backbone_half,
    build_resnet_backbone_minor,
    build_resnet_backbone_half1,
    build_resnet_backbone_half2,
    build_resnet_backbone_half4,
)
from .vit import (
    build_vit_base_backbone,
    build_vit_small_backbone,
    build_vit_tiny_backbone,
)
from .convnext import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    convnext_xlarge,
    convnext_tiny_minor,
    convnext_tiny_half
)
from .patch_tk_ms import build_ptkms_backbone
from .resnet_hybrid import build_resnet5034_backbone

__all__ = [k for k in globals().keys() if not k.startswith("_")]
# TODO can expose more resnet blocks after careful consideration
