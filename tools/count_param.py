import sys

sys.path.append("./")
import fire
import torch
from psd2.config import get_cfg
from psd2.modeling import build_model
from psd2.utils.logger import setup_logger


def main(cfg_file):
    setup_logger(name="psd2")
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    model: torch.nn.Module = build_model(cfg)
    for mn, mm in model.named_children():
        strs = [mn]
        total = 0
        for pn, pv in mm.named_parameters():
            num_p = pv.numel()
            strs.append("\t" + mn + "." + pn + ":\t {}".format(num_p))
            total += num_p
        strs[0] = strs[0] + ":\t {}".format(total)
        show_str = "\n".join(strs)
        print(show_str)


if __name__ == "__main__":
    fire.Fire(main)
