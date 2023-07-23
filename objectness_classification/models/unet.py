import lightning.pytorch as pl
import torch
import torch.optim as optim
import segmentation_models_pytorch as smp


class System(pl.LightningModule):
    def __init__(
        self,
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        out_channels=1,  # [1, 2]
        mode="binary",  # ["binary", "multilabel"]
        threshold=0.5,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            # activation="sigmoid",
        )
        self.lossfn = self._set_lossfn()
        self.optimizer = self._set_optimizers()

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, phase="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._shared_step(batch, phase="val")

    def test_step(self, batch, batch_idx, dataloader_idx):
        _ = self._shared_step(batch, phase="test")

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        out = self.model(x)
        return out > self.hparams.threshold

    def _set_lossfn(self):
        lossfn = smp.losses.DiceLoss(
            mode=self.hparams.mode,
            from_logits=True,
        )

        self.hparams.update({"lossfn": lossfn.__class__.__name__})
        return lossfn

    def _set_optimizers(self):
        opts = {
            "optimizer": (optimizer := optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
            )),
            "lr_scheduler": optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(range(2,10,2)),
                gamma=0.1,
            ),
        }

        self.hparams.update({k:v.__class__.__name__ for k,v in opts.items()})
        return opts

    def _shared_step(self, batch, phase="train"):
        frames, labels = batch[0], batch[1]
        logits = self.model(frames)

        loss = self.lossfn(logits, labels[:,0:self.hparams.out_channels,:,:])
        metric_dict = self._calc_metric(logits, labels)

        self.log(f"loss/{phase}", loss, on_step=True, on_epoch=True)
        metric_dict = {f"{k}/{phase}":v for k,v in metric_dict.items()}
        self.log_dict(metric_dict, on_step=True, on_epoch=True)

        return loss

    def _calc_metric(self, output, target):
        obj_mask, hand_mask = target[:,0:1,:,:], target[:,1:2,:,:]
        output = output[:,0:1,:,:]

        tp, fp, fn, tn = smp.metrics.get_stats(
            output,
            obj_mask.to(torch.int64),
            mode=self.hparams.mode,
            threshold=self.hparams.threshold,
        )

        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2 = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

        convergence = (output.sigmoid() * hand_mask).sum() / hand_mask.sum()

        return {
            f"iou_score": iou,
            f"f1_score": f1,
            f"f2_score": f2,
            f"accuracy": acc,
            f"recall": recall,
            f"convergence": convergence,
        }
