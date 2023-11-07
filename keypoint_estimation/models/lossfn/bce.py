import torch.nn as nn
import torch.nn.functional as nnf

from lossfn import create_soft_label


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        loss = nnf.binary_cross_entropy_with_logits(input, target)
        return loss


class SoftBCELoss(nn.Module):
    def __init__(self, smooth_type):
        super().__init__()
        self.smooth_type = smooth_type

    def forward(self, input, target):
        soft_label = create_soft_label(target, self.smooth_type)
        loss = nnf.binary_cross_entropy_with_logits(input, soft_label)
        return loss
