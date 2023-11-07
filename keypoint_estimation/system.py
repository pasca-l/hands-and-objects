import torch.optim as optim
import lightning as L

from models import set_model
from models.lossfn import set_lossfn
from models.metrics import set_metrics, set_meta_metrics


class KeypointEstModule(L.LightningModule):
    def __init__(
        self,
        model_name="vivit",
        lossfn_name="asyml",
        frame_num=16,
        lr=1e-4,
        mode="multilabel",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = set_model(
            model_name,
            out_channel=frame_num,
        )

        self.lossfn = set_lossfn(lossfn_name)
        self.optimizer = self._set_optimizers()

        self.metrics = set_metrics(mode=mode, frame_num=frame_num)
        self.meta_metric = set_meta_metrics()

        self.hparams.update({"model": self.model.__class__.__name__})
        self.hparams.update({"lossfn": self.lossfn.__class__.__name__})
        self.hparams.update(
            {k: v.__class__.__name__ for k, v in self.optimizer.items()}
        )

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, phase="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._shared_step(batch, phase="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        _ = self._shared_step(batch, phase="test")

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        out = self.model(x)
        return out

    def _set_optimizers(self):
        opts = {
            "optimizer": (optimizer := optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.lr,
            )),
            "lr_scheduler": optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=10,
            ),
        }
        return opts

    def _shared_step(self, batch, phase="train"):
        # frames: torch.Size([b, frame_num, ch, w, h])
        # labels: torch.Size([b, frame_num])
        frames, labels = batch[0], batch[1]
        frames = frames.float()
        labels = labels.float()

        # expected frames: torch.Size([b, frame_num, ch, w, h]) as double
        logits = self.model(frames)

        # expected labels: torch.Size([b, frame_num]) as floating point
        # expected logits: torch.Size([b, frame_num])
        loss = self.lossfn(logits, labels)

        # metalabels: info.select("sample_pnr_diff")
        metalabels = batch[2]
        metrics = self._calc_metrics(logits, labels, metalabels)

        self.log(f"loss/{phase}", loss, on_step=False, on_epoch=True)
        metric_dict = {f"{k}/{phase}":v for k,v in metrics.items()}
        self.log_dict(metric_dict, on_step=False, on_epoch=True)

        return loss

    def _calc_metrics(self, output, target, metalabel):
        metrics = self.metrics(output, target)
        meta_metrics = self.meta_metrics(output, metalabel)

        return metrics | meta_metrics
