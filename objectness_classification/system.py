import pytorch_lightning as pl


class ObjnessClassifier(pl.LightningModule):
    def __init__(
        self,
        model=None,
        loss=None,
        optimizer=None,
        metric=None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric

    def _shared_step(self, batch, phase="train"):
        frames, labels = batch[0], batch[1]
        logits = self.model(frames)
        loss = self.loss(logits, labels)
        metric_dict = self.metric(logits, labels)

        self.log(f"loss/{phase}", loss, on_step=True, on_epoch=True)
        metric_dict = {f"{k}/{phase}":v for k,v in metric_dict.items()}
        self.log_dict(metric_dict, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, phase="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._shared_step(batch, phase="val")

    def test_step(self, batch, batch_idx):
        _ = self._shared_step(batch, phase="test")

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        return self.model(x)
