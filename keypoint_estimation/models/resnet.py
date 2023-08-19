import lightning as L
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torchmetrics as tm


class System(L.LightningModule):
    def __init__(self, out_channels=2, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = tv.models.resnet50(
            weights=tv.models.ResNet50_Weights.DEFAULT
        )
        self.model.fc = nn.Linear(2048, out_channels)

        self.lossfn = self._set_lossfn()
        self.optimizer = self._set_optimizers()

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, phase="train")

    def validation_step(self, batch, batch_idx):
        _ = self._shared_step(batch, phase="val")

    def test_step(self, batch, batch_idx, dataloader_idx):
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
            "optimizer": optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
            )
        }

        self.hparams.update({k:v.__class__.__name__ for k,v in opts.items()})
        return opts

    def _shared_step(self, batch, phase="train"):
        frames, labels = batch[0], batch[1]

        frames = frames[:,0,:,:,:].permute((0,3,1,2)).float()
        logits = self.model(frames)

        loss = self.lossfn(logits, labels)
        # metrics = self._calc_metrics(logits, labels)

        self.log(f"loss/{phase}", loss, on_step=True, on_epoch=True)
        # metric_dict = {f"{k}/{phase}":v for k,v in metrics.items()}
        # self.log_dict(metric_dict, on_step=True, on_epoch=True)

    def _calc_metrics(self, output, target):
        accuracy = tm.functional.accuracy(output, target)

        return {
            "accuracy": accuracy,
        }


if __name__ == "__main__":
    system = System()
    print(system.model)
