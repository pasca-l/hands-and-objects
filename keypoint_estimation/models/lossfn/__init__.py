import numpy as np
import torch
import segmentation_models_pytorch as smp


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

        soft_label = torch.tensor(
            np.array([
                np.convolve(label[i].cpu(), gauss, mode="same")
                for i in range(batch)
            ]),
            device=torch.device(label.device),
            dtype=torch.float32,
        )

    else:
        raise Exception(f"No smooth_type named: {smooth_type}")

    return soft_label


from .asymmetric_loss import AsymmetricLoss, SoftAsymmetricLoss
from .bce import BCELoss, SoftBCELoss
from .ce import CELoss, SoftCELoss
from .mse import MSELoss, SoftMSELoss, RMSELoss, SoftRMSELoss


def set_lossfn(name, mode="multilabel", smooth_type="gauss"):
    if name == "asyml":
        return AsymmetricLoss()

    elif name == "softasyml":
        return SoftAsymmetricLoss(
            smooth_type=smooth_type,
        )

    elif name == "bce":
        return BCELoss()

    elif name == "softbce":
        return SoftBCELoss(
            smooth_type=smooth_type,
        )

    elif name == "ce":
        return CELoss()

    elif name == "softce":
        return SoftCELoss(
            smooth_type=smooth_type,
        )

    elif name == "mse":
        return MSELoss()

    elif name == "softmse":
        return SoftMSELoss(
            smooth_type=smooth_type,
        )

    elif name == "rmse":
        return RMSELoss()

    elif name == "softrmse":
        return SoftRMSELoss(
            smooth_type=smooth_type,
        )

    elif name == "dice":
        return smp.losses.DiceLoss(
            mode=mode,
            from_logits=True,
        )

    elif name == "focal":
        return smp.losses.FocalLoss(
            mode=mode,
            alpha=0.02, # prior probability of having positive value in target
        )

    else:
        raise Exception(f"No lossfn named: {name}")
