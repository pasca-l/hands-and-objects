import torch
import torch.nn as nn


class I3DResNet(nn.Module):
    def __init__(
        self,
        num_out_frames=16,
    ):
        super().__init__()

        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "i3d_r50",
            pretrained=True,
        )
        self.model.blocks[6].proj = nn.Linear(
            in_features=2048, out_features=num_out_frames, bias=True
        )

    def forward(self, x):
        # permute torch.Size([b, frame_num, ch, w, h])
        # to torch.Size([b, ch, frame_num, w, h])
        x = x.permute(0,2,1,3,4)
        x = self.model(x)
        return x
