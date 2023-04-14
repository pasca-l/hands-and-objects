import pytorch_lightning as pl
# from torchmetrics.classification import BinaryJaccardIndex


class ObjnessClassifier(pl.LightningModule):
    def __init__(self, sys):
        super().__init__()
        self.model = sys.model
        self.loss = sys.loss
        self.optimizer = sys.optimizer
        self.metric = sys.metric

    def training_step(self, batch, batch_idx):
        frames, labels = batch[0], batch[1]
        logits = self.model(frames)
        loss = self.loss(logits, labels)

        iou, f1, f2, acc, recall = self.metric(logits, labels)

        self.log("train_loss", loss, on_step=True)
        self.log_dict(
            {
                "iou": iou,
                "accuracy": acc,
            },
            on_step=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        frames, labels = batch[0], batch[1]
        logits = self.model(frames)
        loss = self.loss(logits, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        return self.model(x)
