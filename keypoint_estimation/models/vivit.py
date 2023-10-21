import torch.nn as nn
import torch.optim as optim
import torchmetrics
import lightning as L
import transformers


class System(L.LightningModule):
    def __init__(
        self,
        frame_num=16,
        lr=1e-4,
        mode="multilabel",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ViViT(
            out_channel=frame_num,
        )

        self.lossfn = self._set_lossfn()
        self.optimizer = self._set_optimizers()

        self.stats = torchmetrics.StatScores(task=mode, num_labels=frame_num)
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task=mode, num_labels=frame_num),
            torchmetrics.Precision(task=mode, num_labels=frame_num),
            torchmetrics.Recall(task=mode, num_labels=frame_num),
            torchmetrics.F1Score(task=mode, num_labels=frame_num),
        ])

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

    def _set_lossfn(self):
        lossfn = nn.BCEWithLogitsLoss()
        return lossfn

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

        self.log(f"loss/{phase}", loss, on_step=True, on_epoch=True)
        metric_dict = {f"{k}/{phase}":v for k,v in metrics.items()}
        self.log_dict(metric_dict, on_step=True, on_epoch=True)

        return loss

    def _calc_metrics(self, output, target, metalabel):
        tp, fp, tn, fn, sup = self.stats(output, target)
        stat_dict = {
            "TruePositives": tp,
            "FalsePositives": fp,
            "TrueNegatives": tn,
            "FalseNegatives": fn,
        }

        # preds outside of [0,1] will be considered as logits,
        # and sigmoid() is auto applied
        metrics = self.metrics(output, target)

        # metalabel contains the nearest temporal error,
        # so relevant values are summed
        preds = output.sigmoid() > 0.5
        temp_err = (preds * metalabel).sum() / preds.sum() \
                   if preds.sum() > 0 else 0.0
        meta_metrics = {
            "AverageNearestKeyframeError": temp_err,
        }

        return stat_dict | metrics | meta_metrics


class ViViT(nn.Module):
    def __init__(
        self,
        out_channel,
    ):
        super().__init__()

        config = transformers.VivitConfig(
            num_frames=out_channel,
        )
        self.vivit = transformers.VivitModel(config)
        self.fc_norm = nn.LayerNorm((768,))
        self.classifier = nn.Linear(in_features=768, out_features=out_channel)

    def forward(self, x):
        x = self.vivit(x).last_hidden_state
        x = self.fc_norm(x.mean(1))
        x = self.classifier(x)

        return x
