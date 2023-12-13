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

    elif name == "bceanddice":
        return BCEAndDiceLoss(
            mode=mode
        )

    else:
        raise Exception(f"No lossfn named: {name}")


class BCEAndDiceLoss(nn.Module):
    def __init__(self, mode):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(
            mode=mode,
            from_logits=True,
        )

    def forward(self, input, target):
        bce_loss = self.bce_loss(input, target)
        dice_loss = self.dice_loss(input, target)

        return bce_loss * 0.5 + dice_loss * 0.5
