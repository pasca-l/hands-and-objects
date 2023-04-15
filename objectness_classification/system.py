import pytorch_lightning as pl


class ObjnessClassifier(pl.LightningModule):
    def __init__(
        self,
        model,
        loss,
        optimizer,
        metric=None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric

    def training_step(self, batch, batch_idx):
        frames, labels = batch[0], batch[1]
        logits = self.model(frames)
        loss = self.loss(logits, labels)

        self.log("train_loss", loss, on_step=True)

        if self.metric is not None:
            metric_dict = self.metric(logits, labels)
            self.log_dict(metric_dict, on_step=True)

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
