import sys

sys.path.append("./")
import fire
import torch
from psd2.checkpoint import DetectionCheckpointer as Checkpointer
from psd2.config import get_cfg
from psd2.modeling import build_model
from psd2.utils.events import EventStorage, TensorboardXWriter, get_event_storage
from psd2.utils.logger import setup_logger
import os


def build_train_loader(cfg):
    from psd2.data.catalog import MapperCatalog
    from psd2.data.build import build_detection_train_loader

    mapper = MapperCatalog.get(cfg.DATASETS.TRAIN[0])(cfg, True)
    return build_detection_train_loader(cfg, mapper=mapper)


def build_test_loader(cfg, dataset_name):
    from psd2.data.catalog import MapperCatalog
    from psd2.data.build import build_detection_test_loader

    mapper = MapperCatalog.get(dataset_name)(cfg, False)
    return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def setup(config_file, func_name):
    setup_logger(name="psd2")

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.VIS_PERIOD = 1
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "test-model")

    device = torch.device("cuda")
    model = build_model(cfg)
    model.to(device)
    if "train" in func_name:
        data_loader = build_train_loader(cfg)
    elif "test" in func_name:
        data_loader = build_test_loader(cfg=cfg, dataset_name=cfg.DATASETS.TEST[0])
    else:
        raise ValueError("Func Name: {}".format(func_name))

    return cfg, model, data_loader


def test_training(cfg, model, data_loader, writer):
    model.train()
    storage = get_event_storage()

    for batch_inputs in data_loader:
        losses = model(batch_inputs)
        torch.cuda.empty_cache()
        writer.write()
        storage.step()


def test_inference(cfg, model, data_loader, writer):
    model.eval()
    storage = get_event_storage()

    for batch_inputs in data_loader:
        out = model(batch_inputs)
        torch.cuda.empty_cache()
        writer.write()
        storage.step()


def main(cfg_file, func_name, resume=False):
    cfg, model, data_loader = setup(config_file=cfg_file, func_name=func_name)

    Checkpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=resume
    )
    func_map = {
        "train": test_training,
        "test": test_inference,
    }
    writer = TensorboardXWriter(log_dir=cfg.OUTPUT_DIR)

    with EventStorage(start_iter=0):
        func_map[func_name](cfg, model, data_loader, writer)


if __name__ == "__main__":
    fire.Fire(main)
