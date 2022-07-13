import torch
import torch.optim as optim
import torch.nn.functional as nnf
import pytorch_lightning as pl
import torchmetrics


class PNRLocalizer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self.model(x)
        loss = nnf.binary_cross_entropy(logits, y)

        #acc = self.train_acc(torch.argmax(logits, dim=1), y)
        #self.log("train_performance", {"train_loss": loss, "train_acc": acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self.model(x)
        loss = nnf.binary_cross_entropy(logits, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        return optimizer