import torch
import torch.nn as nn
from perceiver_pytorch import Perceiver


class SlowFastPreceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.path_transform = PackPathwayTransform()

        slowfast = torch.hub.load('facebookresearch/pytorchvideo',
                                  'slowfast_r50', pretrained=True)
        backbone = nn.ModuleList([*list(slowfast.blocks.children())])
        self.b1 = backbone[:1]
        self.b2 = backbone[1:-2]
        self.head = backbone[-2:]

        self.perceiver = Perceiver(
            input_channels=80,
            input_axis=3,
            num_freq_bands=6,
            max_freq=10.0,
            depth=3,
            num_latents=256,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            attn_dropout=0.0,
            ff_dropout=0.0,
            weight_tie_layers=True,
            self_per_cross_attn=6
        )

    def forward(self, x):
        x = self.path_transform(x)
        for layer in self.b1:
            x = layer(x)

        x_per = x[0].permute((0,2,3,4,1))
        x_per = self.perceiver(x_per)

        for layer in self.b2:
            x = layer(x)

        x.append(x_per)

        print([i.shape for i in x])

        for layer in self.head:
            x = layer(x)

        print(x.shape)

        return x


class PackPathwayTransform():
    def __init__(self):
        self.alpha = 4

    def __call__(self, frames):
        fast_pathway = frames
        slow_pathway = torch.index_select(frames, 2, 
            torch.linspace(
                0, frames.shape[2] - 1, frames.shape[2] // self.alpha).long())

        return [slow_pathway, fast_pathway]
