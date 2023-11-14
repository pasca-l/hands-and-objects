import torch.nn as nn
import torch.nn.functional as nnf

from . import create_soft_label


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        loss = nnf.cross_entropy(input, target)
        return loss


class SoftCELoss(nn.Module):
    def __init__(self, smooth_type):
        super().__init__()
        self.smooth_type = smooth_type
        self.ce_loss = CELoss()

    def forward(self, input, target):
        soft_label = create_soft_label(target, self.smooth_type)
        loss = self.ce_loss(input, soft_label)
        return loss
