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
        self.save_hyperparameters()

    def _shared_step(self, batch, batch_idx):
        frames, labels = batch[0], batch[1]
        logits = self.model(frames)
        loss = self.loss(logits, labels)
        metric_dict = self.metric(logits, labels)

        return loss, metric_dict

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, batch_idx)

        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric_dict = self._shared_step(batch, batch_idx)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        metric_dict = {f"val_{k}":v for k,v in metric_dict.items()}
        self.log_dict(metric_dict, on_step=True)

    def test_step(self, batch, batch_idx):
        loss, metric_dict = self._shared_step(batch, batch_idx)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        metric_dict = {f"test_{k}":v for k,v in metric_dict.items()}
        self.log_dict(metric_dict, on_step=True)

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        return self.model(x)
