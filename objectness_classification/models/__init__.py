from .baseline import Baseline
from .unet import Unet
from .transunet import TransUNet
from .segmenter import Segmenter


def set_model(name, **kwargs):
    if name == "baseline":
        return Baseline(**kwargs)

    elif name == "unet":
        return Unet(**kwargs)

    elif name == "transunet":
        return TransUNet(**kwargs)

    elif name == "segmenter":
        return Segmenter(**kwargs)

    else:
        raise Exception(f"No model named: {name}")


def adjust_param(param, pretrain_mode):
    renamed_param = {}

    if pretrain_mode == "pretrain":
        pass

    else:
        raise Exception(f"No pretrain mode named: {pretrain_mode}")

    return renamed_param
