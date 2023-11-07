from models.baseline import Baseline
from models.vivit import ViViT


def set_model(name, out_channel):
    if name == "baseline":
        return Baseline()

    elif name == "vivit":
        return ViViT(out_channel)

    else:
        raise Exception(f"No model named: {name}")
