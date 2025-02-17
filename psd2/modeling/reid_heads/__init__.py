# import imp
from .roi_fc_head import RoiFcHead
from .roi_coseg_head import RoiCoSegHead
from .msroi_fc_head import MsRoiFcHead, MsRoiFcHeadSide
from .msroi_coseg_head import MsRoiCoSegHead
from .dt_decoder_head import (
    DTransDecoderHead,
    DTransDecoderHeadClean,
    MscDTransDecoderHead,
    MscDTransDecoderHeadClean,
    DTransDecoderHeadL2P,
    DTransDecoderHeadL2PClean,
    PstrDecoderHead,
    PstrDecoderHeadClean,
    ContrastDecoderHead,
)
from .dt_decoder_x_head import DTransDecoderXHead, DTransDecoderXHeadClean
from .point_head import PointHead
from .base_reid_head import ReidHeadBase
from .bnneck_head import BnneckHead
from .rcnn_heads import Res5OIMHead, Res5NAEHead

name_to_heads = {
    "roi_fc": RoiFcHead,
    "roi_coseg": RoiCoSegHead,
    "msroi_fc": MsRoiFcHead,
    "msroi_fc_side": MsRoiFcHeadSide,
    "msroi_coseg": MsRoiCoSegHead,
    "dt_decoder": DTransDecoderHead,
    "dt_decoder_v": DTransDecoderHeadClean,
    "dt_decoder_l2p": DTransDecoderHeadL2P,
    "dt_decoder_v_l2p": DTransDecoderHeadL2PClean,
    "dt_decoder_x": DTransDecoderXHead,
    "dt_decoder_xv": DTransDecoderXHeadClean,
    "msc_dt_decoder": MscDTransDecoderHead,
    "msc_dt_decoder_v": MscDTransDecoderHeadClean,
    "point": PointHead,
    "bnneck": BnneckHead,
    "res5_oim_head": Res5OIMHead,
    "res5_nae_head": Res5NAEHead,
    "pstr_dt_decoder": PstrDecoderHead,
    "pstr_dt_decoder_v": PstrDecoderHeadClean,
    "contrast_dt_head": ContrastDecoderHead,
}


def build_reid_head(head_cfg):
    return name_to_heads[head_cfg.NAME](head_cfg)


def build_rcnn_roi_reid_heads(cfg, input_shape):
    name = cfg.REID_HEAD.NAME
    return name_to_heads[name](cfg, input_shape)
