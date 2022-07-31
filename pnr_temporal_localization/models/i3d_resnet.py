import numpy as np
import torch, torch.fx
import torch.nn as nn
import torch.optim as optim


class I3DResNetSys():
    def __init__(self):
        self.model = I3DResNet()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )


class I3DResNet(nn.Module):
    def __init__(self, frame_num=32):
        super().__init__()
        self.frame_num = frame_num

        resnet3d = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        resnet3d_modules = nn.ModuleList([*list(resnet3d.blocks.children())])
        self.backbone = nn.Sequential(*resnet3d_modules[:-1])
        self.pool = nn.AvgPool3d(kernel_size=(1, 7, 7), stride=1, padding=0)
        self.drop = nn.Dropout(p=0.5, inplace=False)
        self.proj = nn.Linear(in_features=2048, out_features=1, bias=True)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.backbone(x)

        x = self.pool(x)
        x = self.drop(x)
        x = x.permute((0, 2, 3, 4, 1))
        x = self.proj(x)
        x = x.permute((0, 4, 1, 2, 3))
        x = x.view(batch_size, -1)

        return x