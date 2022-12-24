import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp


class System():
    def __init__(self):
        self.model = Unet()
        self.loss = None
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )
        self.label_transform = lambda batch: batch[1]


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weight="imagenet",
            in_channels=3,
            classes=1
        )

    def forward(self, x):
        x = self.unet(x)

        return x
