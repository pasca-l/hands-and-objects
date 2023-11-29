from .baseline import Baseline
from .vivit import ViViT
from .tsvivit import TSViViT
from .mctvivit import MCTViViT


def set_model(name, **kwargs):
    if name == "baseline":
        return Baseline()

    elif name == "vivit":
        return ViViT(**kwargs)

    elif name == "tsvivit":
        return TSViViT(**kwargs)

    elif name == "mctvivit":
        return MCTViViT(**kwargs)

    else:
        raise Exception(f"No model named: {name}")


def adjust_param(param, pretrain_mode):
    renamed_param = {}

    if pretrain_mode == "videomae_k400":
        for k, v in param["model"].items():
            k = k.replace("encoder.patch_embed.proj.", "embeddings.patch_embeddings.projection.")
            k = k.replace(".proj.", ".value.")
            k = k.replace(".blocks.", ".layer.")
            k = k.replace(".attn.", ".attention.")
            k = k.replace(".norm.", ".layernorm.")
            k = k.replace(".norm1.", ".layernorm_before.")
            k = k.replace(".norm2.", ".layernorm_after.")
            k = k.replace(".q_bias", ".query.bias")
            k = k.replace(".v_bias", ".value.bias")
            k = k.replace(".mlp.fc1.", ".intermediate.dense.")
            k = k.replace(".mlp.fc2.", ".output.dense.")

            renamed_param[f"vivit.{k}"] = v

    elif pretrain_mode == "transunet":
        for k, v in param.items():
            k = k.replace("vit.embeddings.", "different.torchsize.")
            k = k.replace("vit.", "vivit.")

            renamed_param[k] = v

    else:
        raise Exception(f"No pretrain mode named: {pretrain_mode}")

    return renamed_param
