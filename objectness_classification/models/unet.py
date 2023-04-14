import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import segmentation_models_pytorch as smp


class System():
    def __init__(self):
        self.model = smp.Unet(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )
        self.loss = smp.losses.DiceLoss(
            mode="binary",
        )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4,
        )

    def metric(self, output, target):
        tp, fp, fn, tn = smp.metrics.get_stats(
            output,
            target.to(torch.int64),
            mode="binary",
            threshold=0.5,
        )
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2 = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

        return iou, f1, f2, accuracy, recall
