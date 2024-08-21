import fire
import torch
from collections import OrderedDict


def trans_param(fp_path):
    params = torch.load(fp_path)["state_dict"]
    trans_params = OrderedDict()
    for pk, pv in params.items():
        if "backbone" in pk:
            trans_params[pk] = pv
        if "neck" in pk:
            if "extra" in pk:
                nk = pk.replace("neck.extra_convs.0", "input_proj.3")
                nk = nk.replace("conv", "0")
                nk = nk.replace("gn", "1")
                trans_params[nk] = pv
            else:
                nk = pk.replace("neck.convs", "input_proj")
                nk = nk.replace("conv", "0")
                nk = nk.replace("gn", "1")
                trans_params[nk] = pv
        if "level_emb" in pk:
            trans_params["transformer.level_embed"] = pv
        if "encoder1" in pk:
            for li in range(6):
                # attn
                # attention weights
                trans_params[
                    "transformer.encoder.layers.{}.self_attn.attention_weights.weight".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder1.layers.{}.attentions.0.attention_weights.weight".format(
                        li
                    )
                ]
                trans_params[
                    "transformer.encoder.layers.{}.self_attn.attention_weights.bias".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder1.layers.{}.attentions.0.attention_weights.bias".format(
                        li
                    )
                ]
                # output_proj
                trans_params[
                    "transformer.encoder.layers.{}.self_attn.output_proj.weight".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder1.layers.{}.attentions.0.output_proj.weight".format(
                        li
                    )
                ]
                trans_params[
                    "transformer.encoder.layers.{}.self_attn.output_proj.bias".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder1.layers.{}.attentions.0.output_proj.bias".format(
                        li
                    )
                ]
                # value_proj
                trans_params[
                    "transformer.encoder.layers.{}.self_attn.value_proj.weight".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder1.layers.{}.attentions.0.value_proj.weight".format(
                        li
                    )
                ]
                trans_params[
                    "transformer.encoder.layers.{}.self_attn.value_proj.bias".format(li)
                ] = params[
                    "bbox_head.transformer.encoder1.layers.{}.attentions.0.value_proj.bias".format(
                        li
                    )
                ]
                # sampling_offsets
                trans_params[
                    "transformer.encoder.layers.{}.self_attn.sampling_offsets.weight".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder1.layers.{}.attentions.0.sampling_offsets.weight".format(
                        li
                    )
                ]
                trans_params[
                    "transformer.encoder.layers.{}.self_attn.sampling_offsets.bias".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder1.layers.{}.attentions.0.sampling_offsets.bias".format(
                        li
                    )
                ]
                # ffn
                fi = 1
                trans_params[
                    "transformer.encoder.layers.{}.linear{}.weight".format(li, fi)
                ] = params[
                    "bbox_head.transformer.encoder1.layers.{}.ffns.0.layers.{}.0.weight".format(
                        li, fi - 1
                    )
                ]
                trans_params[
                    "transformer.encoder.layers.{}.linear{}.bias".format(li, fi)
                ] = params[
                    "bbox_head.transformer.encoder1.layers.{}.ffns.0.layers.{}.0.bias".format(
                        li, fi - 1
                    )
                ]
                fi = 2
                trans_params[
                    "transformer.encoder.layers.{}.linear{}.weight".format(li, fi)
                ] = params[
                    "bbox_head.transformer.encoder1.layers.{}.ffns.0.layers.{}.weight".format(
                        li, fi - 1
                    )
                ]
                trans_params[
                    "transformer.encoder.layers.{}.linear{}.bias".format(li, fi)
                ] = params[
                    "bbox_head.transformer.encoder1.layers.{}.ffns.0.layers.{}.bias".format(
                        li, fi - 1
                    )
                ]
                # norm
                for ni in [1, 2]:
                    trans_params[
                        "transformer.encoder.layers.{}.norm{}.weight".format(li, ni)
                    ] = params[
                        "bbox_head.transformer.encoder1.layers.{}.norms.{}.weight".format(
                            li, ni - 1
                        )
                    ]
                    trans_params[
                        "transformer.encoder.layers.{}.norm{}.bias".format(li, ni)
                    ] = params[
                        "bbox_head.transformer.encoder1.layers.{}.norms.{}.bias".format(
                            li, ni - 1
                        )
                    ]
        if "encoder2" in pk:
            for li in range(6):
                # attn
                # attention weights
                trans_params[
                    "transformer.enc_head.layers.{}.self_attn.attention_weights.weight".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder2.layers.{}.attentions.0.attention_weights.weight".format(
                        li
                    )
                ]
                trans_params[
                    "transformer.enc_head.layers.{}.self_attn.attention_weights.bias".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder2.layers.{}.attentions.0.attention_weights.bias".format(
                        li
                    )
                ]
                # output_proj
                trans_params[
                    "transformer.enc_head.layers.{}.self_attn.output_proj.weight".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder2.layers.{}.attentions.0.output_proj.weight".format(
                        li
                    )
                ]
                trans_params[
                    "transformer.enc_head.layers.{}.self_attn.output_proj.bias".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder2.layers.{}.attentions.0.output_proj.bias".format(
                        li
                    )
                ]
                # value_proj
                trans_params[
                    "transformer.enc_head.layers.{}.self_attn.value_proj.weight".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder2.layers.{}.attentions.0.value_proj.weight".format(
                        li
                    )
                ]
                trans_params[
                    "transformer.enc_head.layers.{}.self_attn.value_proj.bias".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder2.layers.{}.attentions.0.value_proj.bias".format(
                        li
                    )
                ]
                # sampling_offsets
                trans_params[
                    "transformer.enc_head.layers.{}.self_attn.sampling_offsets.weight".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder2.layers.{}.attentions.0.sampling_offsets.weight".format(
                        li
                    )
                ]
                trans_params[
                    "transformer.enc_head.layers.{}.self_attn.sampling_offsets.bias".format(
                        li
                    )
                ] = params[
                    "bbox_head.transformer.encoder2.layers.{}.attentions.0.sampling_offsets.bias".format(
                        li
                    )
                ]
                # ffn
                fi = 1
                trans_params[
                    "transformer.enc_head.layers.{}.linear{}.weight".format(li, fi)
                ] = params[
                    "bbox_head.transformer.encoder2.layers.{}.ffns.0.layers.{}.0.weight".format(
                        li, fi - 1
                    )
                ]
                trans_params[
                    "transformer.enc_head.layers.{}.linear{}.bias".format(li, fi)
                ] = params[
                    "bbox_head.transformer.encoder2.layers.{}.ffns.0.layers.{}.0.bias".format(
                        li, fi - 1
                    )
                ]
                fi = 2
                trans_params[
                    "transformer.enc_head.layers.{}.linear{}.weight".format(li, fi)
                ] = params[
                    "bbox_head.transformer.encoder2.layers.{}.ffns.0.layers.{}.weight".format(
                        li, fi - 1
                    )
                ]
                trans_params[
                    "transformer.enc_head.layers.{}.linear{}.bias".format(li, fi)
                ] = params[
                    "bbox_head.transformer.encoder2.layers.{}.ffns.0.layers.{}.bias".format(
                        li, fi - 1
                    )
                ]
                # norm
                for ni in [1, 2]:
                    trans_params[
                        "transformer.enc_head.layers.{}.norm{}.weight".format(li, ni)
                    ] = params[
                        "bbox_head.transformer.encoder2.layers.{}.norms.{}.weight".format(
                            li, ni
                        )
                    ]
                    trans_params[
                        "transformer.enc_head.layers.{}.norm{}.bias".format(li, ni)
                    ] = params[
                        "bbox_head.transformer.encoder2.layers.{}.norms.{}.bias".format(
                            li, ni
                        )
                    ]
    torch.save(trans_params, fp_path[:-4] + "_d2.pth")


def to_param(fp_path):
    params = torch.load(fp_path)["state_dict"]
    torch.save(params, fp_path[:-4] + "org.pth")


def trans_bn(fp_path):
    ckpt = torch.load(fp_path)
    params = ckpt["model"]
    np = OrderedDict()
    for pk, pv in params.items():
        if "reid_head.bn_necks" in pk:
            npk = pk.replace("bn_necks.0", "bn_necks.0.org_bn")
            np[npk] = pv
        else:
            np[pk] = pv
    ckpt["model"] = np
    torch.save(ckpt, fp_path)


if __name__ == "__main__":
    fire.Fire()
