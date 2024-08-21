from psd2.data.datasets.cdps import load_cdps
from psd2.evaluation import evaluator
import os

from .. import mappers
from ..catalog import DatasetCatalog, MapperCatalog, MetadataCatalog

from .cuhk_sysu import load_cuhk_sysu
from .cuhk_sysu import subset_names as cuhk_subsets
from .prw import load_prw
from .prw import subsets as prw_subsets
from .ptk21 import load_ptk21
from .coco_ch import load_coco_ch
import copy

# TODO change evaluator type to "query"
def register_cuhk_sysu_all(datadir):
    """for subset in cuhk_subsets:
    name = "CUHK-SYSU_" + subset
    DatasetCatalog.register(
        name, lambda: load_cuhk_sysu(datadir, copy.deepcopy(subset))
    )
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")"""
    name = "CUHK-SYSU_" + "Train"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Train"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    name = "CUHK-SYSU_Crops_" + "Train"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Train"))
    MapperCatalog.register(name, mappers.CuhksysuCropsMapper)
    name = "CUHK-SYSU_FixOrgCrops_" + "Train"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Train"))
    MapperCatalog.register(name, mappers.CuhksysuFixOrgCropsMapper)
    # MetadataCatalog.get(name).set(evaluator_type="det")
    name = "CUHK-SYSU_" + "Gallery"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "CUHK-SYSU_" + "TestG50"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_InfQ_" + "TestG50"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhkSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG100"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG100"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_InfQ_" + "TestG100"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG100"))
    MapperCatalog.register(name, mappers.CuhkSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG500"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG500"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_InfQ_" + "TestG500"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG500"))
    MapperCatalog.register(name, mappers.CuhkSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG1000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG1000"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_InfQ_" + "TestG1000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG1000"))
    MapperCatalog.register(name, mappers.CuhkSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG2000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG2000"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_InfQ_" + "TestG2000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG2000"))
    MapperCatalog.register(name, mappers.CuhkSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG4000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_InfQ_" + "TestG4000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.CuhkSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")


def register_prw_all(datadir):
    """
    # TODO fix incorrect subset
    for subset in prw_subsets:
        name = "PRW_" + subset
        DatasetCatalog.register(name, lambda: load_prw(datadir, copy.deepcopy(subset)))
        MapperCatalog.register(name, mappers.PrwMapper)
        MetadataCatalog.get(name).set(evaluator_type="det")
    """
    name = "PRW_Train"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Train"))
    MapperCatalog.register(name, mappers.PrwMapper)
    name = "PRW_Crops_Train"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Train"))
    MapperCatalog.register(name, mappers.PrwCropsMapper)
    name = "PRW_FixOrgCrops_Train"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Train"))
    MapperCatalog.register(name, mappers.PrwFixOrgCropsMapper)
    # MetadataCatalog.get(name).set(evaluator_type="det")
    name = "PRW_Query"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Query"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PRW_InfQ"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Query"))
    MapperCatalog.register(name, mappers.PrwSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PRW_Gallery"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")


def register_ptk21_all(datadir):
    name = "PoseTrack21_Train"
    DatasetCatalog.register(name, lambda: load_ptk21(datadir, "Train"))
    MapperCatalog.register(name, mappers.Ptk21Mapper)
    # MetadataCatalog.get(name).set(evaluator_type="det")
    name = "PoseTrack21_Query"
    DatasetCatalog.register(name, lambda: load_ptk21(datadir, "Query"))
    MapperCatalog.register(name, mappers.Ptk21Mapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PoseTrack21_InfQ"
    DatasetCatalog.register(name, lambda: load_ptk21(datadir, "Query"))
    MapperCatalog.register(name, mappers.Ptk21SearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PoseTrack21_Gallery"
    DatasetCatalog.register(name, lambda: load_ptk21(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.Ptk21Mapper)
    MetadataCatalog.get(name).set(evaluator_type="det")


def register_cdps_all(datadir):
    name = "CDPS_Train"
    DatasetCatalog.register(name, lambda: load_cdps(datadir, "Train"))
    MapperCatalog.register(name, mappers.CdpsMapper)
    # MetadataCatalog.get(name).set(evaluator_type="det")
    name = "CDPS_Query"
    DatasetCatalog.register(name, lambda: load_cdps(datadir, "Query"))
    MapperCatalog.register(name, mappers.CdpsMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CDPS_InfQ"
    DatasetCatalog.register(name, lambda: load_cdps(datadir, "Query"))
    MapperCatalog.register(name, mappers.CdpsSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CDPS_Gallery"
    DatasetCatalog.register(name, lambda: load_cdps(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CdpsMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")


def register_cococh_all(datadir):
    name = "COCO-CH_Train"
    DatasetCatalog.register(name, lambda: load_coco_ch(datadir, "train"))
    MapperCatalog.register(name, mappers.COCOCHMapper)
    name = "COCO-CH_Val"
    DatasetCatalog.register(name, lambda: load_coco_ch(datadir, "val"))
    MapperCatalog.register(name, mappers.COCOCHMapper)
    name = "COCO-CH_Plain"
    DatasetCatalog.register(name, lambda: load_coco_ch(datadir, "train"))
    MapperCatalog.register(name, mappers.SearchMapper)


def register_cococh2v_all(datadir):
    name = "COCO-CH2v_Train"
    DatasetCatalog.register(name, lambda: load_coco_ch(datadir, "train"))
    MapperCatalog.register(name, mappers.COCOCH2vMapper)
    name = "COCO-CH2v_Val"
    DatasetCatalog.register(name, lambda: load_coco_ch(datadir, "val"))
    MapperCatalog.register(name, mappers.COCOCH2vMapper)


def register_cococh_ncd_all(datadir):
    name = "COCO-CH-ncd_Train"
    DatasetCatalog.register(name, lambda: load_coco_ch(datadir, "train", False))
    MapperCatalog.register(name, mappers.COCOCHMapper)
    name = "COCO-CH-ncd_Val"
    DatasetCatalog.register(name, lambda: load_coco_ch(datadir, "val", False))
    MapperCatalog.register(name, mappers.COCOCHMapper)


_root = os.getenv("PS_DATASETS", "Data/ReID")
register_cuhk_sysu_all(os.path.join(_root, "cuhk_sysu"))
register_prw_all(os.path.join(_root, "PRW"))
# register_cdps_all(os.path.join(_root, "CDPS_mini_v1.1"))
register_cdps_all(os.path.join(_root, "CDPS"))
register_ptk21_all(os.path.join(_root, "PoseTrack21"))
_root_det = "Data/DetData"
register_cococh_all(_root_det)
register_cococh_ncd_all(_root_det)
register_cococh2v_all(_root_det)
