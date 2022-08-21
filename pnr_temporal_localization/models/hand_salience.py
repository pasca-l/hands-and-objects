from turtle import forward
import torch.nn as nn
import torch.optim as optim


class System():
    def __init__(self):
        self.model = HandSalience()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        self.label_transform = IdentityTransform()


class HandSalience(nn.Module):
    def __init__(self):
        super().__init__()
        self.hand_detect = HandDetect()

    def forward(self, x):
        x = self.hand_detect(x)

        return x


class HandDetect():
    def __init__(self):
        pass

    def __call__(self, frames):
        frame_num = frames.shape[2]


class IdentityTransform():
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch[1]