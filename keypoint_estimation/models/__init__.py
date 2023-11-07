from lossfn import *
from metrics import *

from .baseline import Baseline
from .vivit import ViViT


def set_model(name, out_channel):
    if name == "baseline":
        return Baseline()

    elif name == "vivit":
        return ViViT(out_channel)

    else:
        raise Exception(f"No model named: {name}")
