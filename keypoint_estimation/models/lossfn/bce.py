import torch.nn as nn
import torch.nn.functional as nnf

from . import create_soft_label


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        loss = nnf.binary_cross_entropy_with_logits(input, target.float())
        return loss


class SoftBCELoss(nn.Module):
    def __init__(self, smooth_type):
        super().__init__()
        self.smooth_type = smooth_type
        self.bce_loss = BCELoss()

    def forward(self, input, target):
        soft_label = create_soft_label(target, self.smooth_type)
        loss = self.bce_loss(input, soft_label)
        return loss
