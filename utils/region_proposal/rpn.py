import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np

# reference from https://towardsdatascience.com/everything-about-fasterrcnn-6d758f5a6d79


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def main():
    image_path = "../../objectness_classification/test_data/3dead068-318a-49c5-8036-e176e50cbf50/pnr_503.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))

    trans = transforms.ToTensor()
    img = trans(image)
    img = img.unsqueeze(0)

    fasterrcnn = models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True
    )
    # fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(
    #     fasterrcnn.roi_heads.box_predictor.cls_score.in_features,
    #     2
    # )

    fasterrcnn.eval()
    out = fasterrcnn(img)

    for box in out[0]['boxes']:
        cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])), (255,0,0), thickness=1)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./rpn.png", image)


if __name__ == '__main__':
    main()
