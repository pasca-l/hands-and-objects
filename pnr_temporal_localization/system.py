import torch.optim as optim
import torch.nn.functional as nnf
import pytorch_lightning as pl


class PNRLocalizer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1].float()
        y_hat = self.model(x)
        loss = nnf.binary_cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1].float()
        y_hat = self.model(x)
        loss = nnf.binary_cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        return optimizer