import torch
import torch.optim as optim
import segmentation_models_pytorch as smp


class System():
    def __init__(self):
        self.model = smp.Unet(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            # activation="sigmoid",
        )
        self.loss = smp.losses.DiceLoss(
            mode="binary",
            from_logits=True,
        )
        self.optimizer = {
            "optimizer": (optimizer := optim.Adam(
                self.model.parameters(),
                lr=1e-4,
            )),
            "lr_scheduler": optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[2, 4, 6, 8],
                gamma=0.1,
            ),
        }

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
        acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

        return {
            "iou_score": iou,
            "f1_score": f1,
            "f2_score": f2,
            "accuracy": acc,
            "recall": recall,
        }
