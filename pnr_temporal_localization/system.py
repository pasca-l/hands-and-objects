import importlib

import pytorch_lightning as pl
# import torchmetrics


class PNRLocalizer(pl.LightningModule):
    def __init__(self, sys_name):
        super().__init__()
        module = importlib.import_module(f'models.{sys_name}')
        system = module.System()

        self.model = system.model
        self.loss = system.loss
        self.optimizer = system.optimizer

        if sys_name == 'bmn':
            self.label_function = module.LabelTransform()
        else:
            self.label_function = IdentityTransform()

        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()

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

    def configure_optimizers(self):
        return self.optimizer


class IdentityTransform():
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch[1]