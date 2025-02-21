import os

from .. import mappers
from ..catalog import DatasetCatalog, MapperCatalog, MetadataCatalog

from .cuhk_sysu import load_cuhk_sysu
from .prw import load_prw


def register_cuhk_sysu_all(datadir):
    name = "CUHK-SYSU_" + "Train"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Train"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)

    name = "CUHK-SYSU_" + "Gallery"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")

    name = "CUHK-SYSU_InfQ_" + "TestG50"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhkSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")

    name = "CUHK-SYSU_InfQ_" + "TestG100"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG100"))
    MapperCatalog.register(name, mappers.CuhkSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")

    name = "CUHK-SYSU_InfQ_" + "TestG500"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG500"))
    MapperCatalog.register(name, mappers.CuhkSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")

    name = "CUHK-SYSU_InfQ_" + "TestG1000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG1000"))
    MapperCatalog.register(name, mappers.CuhkSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")

    name = "CUHK-SYSU_InfQ_" + "TestG2000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG2000"))
    MapperCatalog.register(name, mappers.CuhkSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")

    name = "CUHK-SYSU_InfQ_" + "TestG4000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.CuhkSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")


def register_prw_all(datadir):

    name = "PRW_Train"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Train"))
    MapperCatalog.register(name, mappers.PrwMapper)

    name = "PRW_InfQ"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Query"))
    MapperCatalog.register(name, mappers.PrwSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PRW_Gallery"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")




_root = os.getenv("PS_DATASETS", "Data/ReID")
register_cuhk_sysu_all(os.path.join(_root, "cuhk_sysu"))
register_prw_all(os.path.join(_root, "PRW"))
