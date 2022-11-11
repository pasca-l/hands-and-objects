import torch.nn as nn
import torch.optim as optim
from i3d_resnet import I3DResNet


class System():
    def __init__(self):
        self.model = HandSalience()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        self.label_transform = lambda batch: batch[1]


class HandSalience(I3DResNet):
    def __init__(self, frame_num=32):
        super().__init__(frame_num)
        self.backbone[0].conv =\
            nn.Conv3d(4, 64, kernel_size=(1,7,7), stride=(1,2,2),
                      padding=(0,3,3), bias=False)
