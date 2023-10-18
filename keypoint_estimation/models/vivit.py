import torch.nn as nn
import torch.nn.functional as nnf
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

        self.model = ViViT(
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
        lossfn = nn.BCEWithLogitsLoss()

        self.hparams.update({"lossfn": lossfn.__class__.__name__})
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

        self.hparams.update({k:v.__class__.__name__ for k,v in opts.items()})
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
        metrics = self._calc_metrics(logits, labels)

        self.log(f"loss/{phase}", loss, on_step=True, on_epoch=True)
        metric_dict = {f"{k}/{phase}":v for k,v in metrics.items()}
        self.log_dict(metric_dict, on_step=True, on_epoch=True)

        return loss

    def _calc_metrics(self, output, target):
        prob = nnf.softmax(output, dim=0)
        preds = (prob > self.hparams.threshold).int()

        accuracy = torch.sum(preds == target) / torch.numel(preds)

        return {
            "accuracy": accuracy,
        }


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
