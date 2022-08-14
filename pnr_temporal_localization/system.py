import pytorch_lightning as pl
# import torchmetrics


class PNRLocalizer(pl.LightningModule):
    def __init__(self, sys):
        super().__init__()
        self.model = sys.model
        self.loss = sys.loss
        self.optimizer = sys.optimizer
        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx):
        frames, label = batch[0], batch[1]
        logits = self.model(frames)
        loss = self.loss(logits, label)

        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frames, label = batch[0], batch[1]
        logits = self.model(frames)
        loss = self.loss(logits, label)

        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer