from .baseline import Baseline
from .unet import Unet
from .transunet import TransUNet
from .setr import SETR


def set_model(name, **kwargs):
    if name == "baseline":
        return Baseline(**kwargs)

    elif name == "unet":
        return Unet(**kwargs)

    elif name == "transunet":
        return TransUNet(**kwargs)

    elif name == "setr":
        return SETR(**kwargs)

    else:
        raise Exception(f"No model named: {name}")


def adjust_param(param, pretrain_mode):
    renamed_param = {}

    if pretrain_mode == "vit-imagenet21k":
        for k, v in param.items():
            if "encoder." in k:
                k = k.replace(".layer.", ".encoder.")

                if ".layernorm" in k:
                    k = k.replace(".layernorm", ".ln")
                elif ".intermediate." in k:
                    k = k.replace(".intermediate.dense.", ".mlp.0.")
                elif ".output." in k:
                    k = k.replace(".output.dense.", ".mlp.3.")

                elif ".attention." in k:
                    if ".key." in k:
                        k = k.replace(
                            ".attention.attention.key.", ".mhsa.w_key."
                        )
                    elif ".query." in k:
                        k = k.replace(
                            ".attention.attention.query.", ".mhsa.w_query."
                        )
                    elif ".value." in k:
                        k = k.replace(
                            ".attention.attention.value.", ".mhsa.w_value."
                        )
                    elif ".output." in k:
                        k = k.replace(
                            ".attention.output.dense.", ".mhsa.w_output.0."
                        )

            elif "embeddings." in k:
                k = f"encoder.{k}"

                if "position_embeddings" in k:
                    k = k.replace(".position_embeddings", "pos_embed")
                elif "patch_embeddings" in k:
                    k = k.replace(".patch_embeddings.projection.", ".")

    else:
        raise Exception(f"No pretrain mode named: {pretrain_mode}")

    return renamed_param
