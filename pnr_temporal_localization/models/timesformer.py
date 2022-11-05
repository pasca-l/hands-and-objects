import torch
import torch.nn as nn
import torch.optim as optim
from timesformer_pytorch import TimeSformer


class System():
    def __init__(self):
        self.model = TimeSformer()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        self.label_transform = IdentityTransform()


class TimeSformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.timesformer = TimeSformer(
            dim = 512,
            image_size = 224,
            patch_size = 16,
            num_frames = 32,
            num_classes = 32,
            depth = 12,
            heads = 8,
            dim_head =  64,
            attn_dropout = 0.1,
            ff_dropout = 0.1
        )

    def forward(self, x):
        x = x.permute((0,2,1,3,4))
        x = self.timesformer(x)

        return x


class IdentityTransform():
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch[1]