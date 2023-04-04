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
        frames, label = batch[0], batch[1]
        logits = self.model(frames)
        loss = self.loss(logits, label)

        iou_score, f1_score, f2_score, accuracy, recall = self.metric(logits, label)

        self.log("train_loss", loss, on_step=True)
        self.log("iou", iou_score, on_step=True)
        self.log("accuracy", accuracy, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frames, label = batch[0], batch[1]
        logits = self.model(frames)
        loss = self.loss(logits, label)

        # metric = BinaryJaccardIndex()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        # self.log("iou", metric(logits, label[:,1:2,:,:]), on_step=True)

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        self.model.eval()
        return self.model(x)
