import torch.nn as nn
import torch.nn.functional as nnf
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
    def __init__(self):
        pass

    def forward(self, input, target):
        bce_loss = nnf.binary_cross_entropy_with_logits(input, target)

        input = input.contiguous()
        target = target.contiguous()
        intersection = (input * target).sum(dim=2).sum(dim=2)
        dice_loss = (1 - (
            (2. * intersection) / (input.sum(dim=2).sum(dim=2) + \
            target.sum(dim=2).sum(dim=2))
        )).mean()

        return bce_loss * 0.5 + dice_loss * 0.5
