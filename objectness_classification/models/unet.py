import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import segmentation_models_pytorch as smp


class System():
    def __init__(self):
        self.model = Unet()
        self.loss = smp.losses.DiceLoss(
            mode='binary'
        )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

    def forward(self, x):
        x = self.unet(x)

        return x


# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super().__init__()

#     def forward(self, inputs, targets, smooth=1):
#         targets = targets[:,1:,:,:]
#         inputs = nnf.sigmoid(inputs)

#         inputs = torch.flatten(inputs)
#         targets = torch.flatten(targets)

#         intersection = (inputs * targets).sum()
#         dice = (2.*intersection + smooth) / \
#                (inputs.sum() + targets.sum() + smooth)

#         return 1 - dice
