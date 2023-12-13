import torch.nn as nn
import segmentation_models_pytorch as smp


class Unet(nn.Module):
    def __init__(
        self,
        encoder_name="resnet101",
        encoder_weights="imagenet",
        unet_in_channels=3,
        num_out_frames=1,
        with_attention=False,
    ):
        super().__init__()

        self.with_attention = with_attention

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=unet_in_channels,
            classes=num_out_frames,
        )

    def forward(self, x):
        x = self.model(x)
        return x
