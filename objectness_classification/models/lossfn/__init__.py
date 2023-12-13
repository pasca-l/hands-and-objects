import torch.nn as nn
import segmentation_models_pytorch as smp


def set_lossfn(name, mode="multilabel"):
    if name == "dice":
        return smp.losses.DiceLoss(
            mode=mode,
            from_logits=True,
        )

    elif name == "bce":
        return nn.BCEWithLogitsLoss()

    elif name == "ce":
        return nn.CrossEntropyLoss()

    else:
        raise Exception(f"No lossfn named: {name}")

