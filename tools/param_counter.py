import fire
import torch

valid_words = ["weight", "bias", "running_mean", "running_var"]


def count_param(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if "model" in sd:
        sd = sd["model"]
    count = 0
    count_names = []
    skip_names = []
    for pn, pv in sd.items():
        is_vp = False
        for vw in valid_words:
            if vw in pn:
                count += pv.view(-1).size()[0]
                count_names.append(pn)
                is_vp = True
                break
        if not is_vp:
            skip_names.append(pn)
    print("Counted Params:")
    print("\n".join(count_names))
    print("Skipped Params:")
    print("\n".join(skip_names))
    print("Total Params: {}".format(count))


if __name__ == "__main__":
    fire.Fire(count_param)
