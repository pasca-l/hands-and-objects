import torch.nn as nn
import torch.optim as optim
import lightning as L
import transformers


class System(L.LightningModule):
    def __init__(
        self,
        frame_num=16,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VideoMAE(
            out_channel=frame_num,
        )

        self.lossfn = self._set_lossfn()
        self.optimizer = self._set_optimizers()

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
        lossfn = nn.CrossEntropyLoss()

        self.hparams.update({"lossfn": lossfn.__class__.__name__})
        return lossfn

    def _set_optimizers(self):
        opts = {
            "optimizer": optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.lr,
            )
        }

        self.hparams.update({k:v.__class__.__name__ for k,v in opts.items()})
        return opts

    def _shared_step(self, batch, phase="train"):
        # expected input torch.Size([1, 16, 3, 224, 224])
        # expected output torch.Size([1, 16])
        frames, labels = batch[0], batch[1]

        logits = self.model(frames)

        loss = self.lossfn(logits, labels)
        metrics = self._calc_metrics(logits, labels)

        self.log(f"loss/{phase}", loss, on_step=True, on_epoch=True)
        metric_dict = {f"{k}/{phase}":v for k,v in metrics.items()}
        self.log_dict(metric_dict, on_step=True, on_epoch=True)

    def _calc_metrics(self, output, target):
        return {
            "accuracy": 0,
        }


class VideoMAE(nn.Module):
    def __init__(
        self,
        out_channel,
    ):
        super().__init__()

        config = transformers.VideoMAEConfig()
        self.videomae = transformers.VideoMAEModel(config)
        self.fc_norm = nn.LayerNorm((768,))
        self.classifier = nn.Linear(in_features=768, out_features=out_channel)

    def forward(self, x):
        x = self.videomae(x).last_hidden_state
        x = self.fc_norm(x.mean(1))
        x = self.classifier(x)

        return x
