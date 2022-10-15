import torch
import pytorch_lightning as pl
from torchmetrics import Metric


class PNRLocalizer(pl.LightningModule):
    def __init__(self, sys):
        super().__init__()
        self.model = sys.model
        self.loss = sys.loss
        self.optimizer = sys.optimizer
        self.label_function = sys.label_transform

        self.frame_err = AbsoluteTemporalError()

    def training_step(self, batch, batch_idx):
        frames = batch[0]
        label = self.label_function(batch)
        logits = self.model(frames)
        loss = self.loss(logits, label)

        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frames = batch[0]
        label = self.label_function(batch)
        logits = self.model(frames)
        loss = self.loss(logits, label)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        info = batch[-1]
        self.frame_err(logits, label, info)
        self.log("val_err", self.frame_err, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer


class AbsoluteTemporalError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("error", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, preds, target, info):
        batch_size, sample_num = preds.shape[0], preds.shape[1]
        diff = torch.argmax(preds, dim=1) - torch.argmax(target, dim=1)
        frame_error = info["total_frame_num"] / sample_num * torch.abs(diff)
        
        self.error += torch.sum(frame_error)
        self.total += batch_size

    def compute(self):
        return self.error / self.total


class BMNErrorMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("error", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, preds, target, info):
        preds = preds[0]
        target = target[0]
        batch_size, sample_num = preds.shape[0], preds.shape[1]
        diff = torch.argmax(preds[:,0,:,0].reshape(batch_size, -1), dim=1) - torch.argmax(target[:,:,0].reshape(batch_size, -1), dim=1).to(preds.device)
        frame_error = info["total_frame_num"] / sample_num * torch.abs(diff)
        
        self.error += torch.sum(frame_error)
        self.total += batch_size

    def compute(self):
        return self.error / self.total