import torch
import torch.nn as nn


class SlowFastPreceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.path_transform = PackPathwayTransform()
        slowfast = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        self.backbone = nn.ModuleList([*list(slowfast.blocks.children())[:-1]])

    def forward(self, x):
        return self.backbone(self.path_transform(x))


class PackPathwayTransform():
    def __init__(self):
        self.alpha = 4

    def __call__(self, frames):
        fast_pathway = frames
        slow_pathway = torch.index_select(frames, 2, 
            torch.linspace(
                0, frames.shape[2] - 1, frames.shape[2] // self.alpha).long())

        return [slow_pathway, fast_pathway]
