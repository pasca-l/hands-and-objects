from .baseline import Baseline
from .vivit import ViViT
from .tsvivit import TSViViT
from .mctvivit import MCTViViT


def set_model(name, **kwargs):
    if name == "baseline":
        return Baseline(**kwargs)

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
            if "decoder" in k:
                continue

            elif "patch_embed" in k:
                k = k.replace(
                    "encoder.patch_embed.proj.", "embedding.projection."
                )

            elif "blocks" in k:
                k = k.replace("encoder.blocks.", "encoder.")

                if ".norm1." in k:
                    k = k.replace(".norm1.", ".ln_before.")
                elif ".norm2." in k:
                    k = k.replace(".norm2.", ".ln_after.")
                elif ".mlp.fc1." in k:
                    k = k.replace(".mlp.fc1.", ".mlp.0.")
                elif ".mlp.fc2." in k:
                    k = k.replace(".mlp.fc2.", ".mlp.3.")

                elif ".attn." in k:
                    k = k.replace(".attn.", ".mhsa.")

                    if ".proj." in k:
                        k = k.replace(".proj.", ".w_output.0.")
                    elif ".qkv." in k:
                        kq = k.replace(".qkv.", ".w_query.")
                        kk = k.replace(".qkv.", ".w_key.")
                        kv = k.replace(".qkv.", ".w_value.")
                        wq, wk, wv = v.split(split_size=768)
                        renamed_param[f"vivit.{kq}"] = wq
                        renamed_param[f"vivit.{kk}"] = wk
                        renamed_param[f"vivit.{kv}"] = wv
                        continue

            renamed_param[f"vivit.{k}"] = v

    elif pretrain_mode == "transunet":
        for k, v in param.items():
            k = k.replace("vit.embeddings.", "different.torchsize.")
            k = k.replace("vit.", "vivit.")

            renamed_param[k] = v

    else:
        raise Exception(f"No pretrain mode named: {pretrain_mode}")

    return renamed_param
