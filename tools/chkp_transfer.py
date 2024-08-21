import torch
import fire

org = ["reid_head.feat_proj.bias", "reid_head.feat_proj.weight"]
tgt = ["reid_head.feat_proj.linear.bias", "reid_head.feat_proj.linear.weight"]


def sd_transfer(path):
    sd = torch.load(path)
    for org_k, tgt_k in zip(org, tgt):
        param = sd["model"].pop(org_k)
        sd["model"][tgt_k] = param
    torch.save(sd, path)


if __name__ == "__main__":
    fire.Fire()
