import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class System():
    def __init__(self) -> None:
        self.model = FasterRCNN()
        self.loss = None
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )
        self.label_transform = lambda batch: batch[1]


class FasterRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fasterrcnn = models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )

    def forward(self, x):
        x = self.fasterrcnn(x)

        return x
