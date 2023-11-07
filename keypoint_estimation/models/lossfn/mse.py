import torch.nn as nn
import torch.nn.functional as nnf

from . import create_soft_label


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        loss = nnf.mse_loss(input.sigmoid(), target)
        return loss


class SoftMSELoss(nn.Module):
    def __init__(self, smooth_type):
        super().__init__()
        self.smooth_type = smooth_type

    def forward(self, input, target):
        soft_label = create_soft_label(target, self.smooth_type)
        loss = nnf.mse_loss(input.sigmoid(), soft_label)
        return loss
