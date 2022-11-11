import torch
import torch.nn as nn
import torch.optim as optim
from perceiver_pytorch import Perceiver


class System():
    def __init__(self):
        self.model = SlowFastPreceiver()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.004,
            momentum=0.9,
            weight_decay=1e-4,
            dampening=0,
            nesterov=False
        )
        self.label_transform = lambda batch: batch[1]


class SlowFastPreceiver(nn.Module):
    def __init__(self, frame_num=32):
        super().__init__()
        self.frame_num = frame_num
        self.path_transform = PackPathwayTransform()

        slowfast = torch.hub.load('facebookresearch/pytorchvideo',
                                  'slowfast_r50', pretrained=True)
        slowfast_modules = nn.ModuleList([*list(slowfast.blocks.children())])
        self.backbone = slowfast_modules[:-1]
        self.head = ResNetBasicHead(self.frame_num)

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
            num_classes=self.frame_num,
            attn_dropout=0.0,
            ff_dropout=0.0,
            weight_tie_layers=True,
            self_per_cross_attn=6
        )

    def forward(self, x):
        x = self.path_transform(x)
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 0:
                x_per = x[0].permute((0,2,3,4,1))
                x_per = self.perceiver(x_per)
        x = self.head(x)
        x = x + 0.01 * x_per

        return x


class PackPathwayTransform():
    def __init__(self):
        self.alpha = 4

    def __call__(self, frames):
        frame_num = frames.shape[2]
        index = torch.linspace(0, frame_num - 1, frame_num // self.alpha,
                               device=frames.device).long()

        fast_pathway = frames
        slow_pathway = torch.index_select(frames, 2, index)

        return [slow_pathway, fast_pathway]


class ResNetBasicHead(nn.Module):
    def __init__(self, frame_num):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.proj = nn.Linear(in_features=2304, out_features=frame_num,
                              bias=True)
        self.output_pool = nn.AdaptiveAvgPool3d(output_size=1)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.dropout(x)
        x = x.permute((0, 2, 3, 4, 1))
        x = self.proj(x)
        x = x.permute((0, 4, 1, 2, 3))
        x = self.output_pool(x)
        x = x.view(batch_size, -1)

        return x
