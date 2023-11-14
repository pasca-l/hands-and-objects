import torch
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
        self.mse_loss = MSELoss()

    def forward(self, input, target):
        soft_label = create_soft_label(target, self.smooth_type)
        loss = self.mse_loss(input, soft_label)
        return loss


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        eps = 1e-6
        loss = torch.sqrt(nnf.mse_loss(input.sigmoid(), target) + eps)
        return loss


class SoftRMSELoss(nn.Module):
    def __init__(self, smooth_type):
        super().__init__()
        self.smooth_type = smooth_type
        self.rmse_loss = RMSELoss()

    def forward(self, input, target):
        soft_label = create_soft_label(target, self.smooth_type)
        loss = self.rmse_loss(input, soft_label)
        return loss
