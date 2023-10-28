import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import segmentation_models_pytorch as smp


def set_lossfn(name, mode="multilabel"):
    if name == "bce":
        return nn.BCEWithLogitsLoss()

    elif name == "softbce":
        return SoftBCEWithLogitsLoss(
            smooth_type="gauss",
        )

    elif name == "mse":
        return MSEFromLogitsLoss()

    elif name == "softmse":
        return SoftMSEFromLogitsLoss(
            smooth_type="gauss",
        )

    elif name == "dice":
        return smp.DiceLoss(
            mode=mode,
            from_logits=True,
        )

    elif name == "focal":
        return smp.FocalLoss(
            mode=mode,
            alpha=0.02, # prior probability of having positive value in target
        )

    else:
        raise Exception(f"No lossfn named: {name}")


def create_soft_label(
    label,
    smooth_type,
    smooth_factor=0.1,
    mu=0,
    sigma=1,
    amp=1,
):
    if smooth_type == "factor":
        soft_label = (1 - label) * smooth_factor + label * (1 - smooth_factor)

    elif smooth_type == "gauss":
        batch, frame_num = label.shape
        x = np.linspace(-3 * sigma, 3 * sigma, frame_num)
        gauss = amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        soft_label = torch.Tensor(
            [np.convolve(label[i], gauss, mode="same") for i in range(batch)]
        )

    else:
        raise Exception(f"No smooth_type named: {smooth_type}")

    return soft_label


class SoftBCEWithLogitsLoss(nn.Module):
    def __init__(self, smooth_type):
        super().__init__()
        self.smooth_type = smooth_type

    def forward(self, input, target):
        soft_label = create_soft_label(target, self.smooth_type)
        loss = nnf.binary_cross_entropy_with_logits(input, soft_label)
        return loss


class MSEFromLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        loss = nnf.mse_loss(input.sigmoid(), target)
        return loss


class SoftMSEFromLogitsLoss(nn.Module):
    def __init__(self, smooth_type):
        super().__init__()
        self.smooth_type = smooth_type

    def forward(self, input, target):
        soft_label = create_soft_label(target, self.smooth_type)
        loss = nnf.mse_loss(input.sigmoid(), soft_label)
        return loss
