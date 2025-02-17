import sys
sys.path.append("./")
import os
import torch
import torch.nn as nn
import fire
from psd2.config import get_cfg
from psd2.checkpoint import DetectionCheckpointer as Checkpointer
from psd2.modeling import build_model
from psd2.utils.events import EventStorage
from psd2.utils.logger import setup_logger
from fvcore.nn import FlopCountAnalysis
import copy
from psd2.utils.env import seed_all_rng
from psd2.utils.events import EventStorage, get_event_storage


def build_test_loader(cfg):
    from psd2.data.catalog import MapperCatalog
    from psd2.data.build import build_detection_test_loader
    datasetname=cfg.DATASETS.TEST[0]
    mapper = MapperCatalog.get(datasetname)(cfg, False)
    return build_detection_test_loader(cfg, datasetname, mapper=mapper)

def build_train_loader(cfg):
    from psd2.data.catalog import MapperCatalog
    from psd2.data.build import build_detection_train_loader

    mapper = MapperCatalog.get(cfg.DATASETS.TRAIN[0])(cfg, True)
    return build_detection_train_loader(cfg, mapper=mapper)


def setup(config_file):
    setup_logger(name="psd2")

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.TEST.IMS_PER_PROC=1
    # cfg.SEED=2024
    cfg.DATALOADER.ASPECT_RATIO_GROUPING=False
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.INPUT.MIN_SIZE_TRAIN=(900,)
    # cfg.OP_COUNT=True

    seed_all_rng(1024)

    device = torch.device("cuda")
    model = build_model(cfg)
    model.to(device)
    test_loader = build_test_loader(cfg)
    train_loader=build_train_loader(cfg)

    return cfg, model, test_loader,train_loader



def train_count( model, data_loader):
    model.train()
    storage = get_event_storage()
    data_count=0
    op_count=0
    for indata in data_loader:
        if data_count==100:
            break
        with FlopTensorDispatchMode(model) as ftdm:
            out = model(indata)
            torch.cuda.empty_cache()
            storage.step()
            flops=copy.deepcopy(ftdm.flop_counts)
            ftdm.reset()
            total=0
        for k,v in flops[''].items():
            total+=v
        data_count+=1
        op_count+=total
    print("GFLOPs: {:.2f}".format(op_count/(1.0e11)))

@torch.no_grad()
def test_count( model, data_loader):
    model.eval()
    data_count=0
    op_count=0
    for indata in data_loader:
        if data_count==100:
            break
        indata,_=model.preprocess_input(indata)
        flops = FlopCountAnalysis(model, indata.tensor)
        op_count+=flops.total()
        print(flops.total())
        torch.cuda.empty_cache()
        data_count+=1
    print("GFLOPs: {:.2f}".format(op_count/(1.0e11)))

def main(cfg_file, resume=False):
    cfg, model, test_loader, train_loader = setup(config_file=cfg_file)

    Checkpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=resume
    )
    #with EventStorage(start_iter=0):
    #    train_count(model,train_loader)
    test_count(model,test_loader)


if __name__ == "__main__":
    fire.Fire(main)
