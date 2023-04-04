import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import segmentation_models_pytorch as smp


class System():
    def __init__(self):
        self.model = smp.Unet(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=2,
            activation="sigmoid",
        )
        self.loss = smp.losses.DiceLoss(
            mode='binary'
        )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )
