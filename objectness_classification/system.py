import torch
import torch.optim as optim
import lightning as L
import segmentation_models_pytorch as smp

from models import set_model, adjust_param
from models.lossfn import set_lossfn


class ObjnessClsModule(L.LightningModule):
    def __init__(
        self,
        model_name="unet",
        pretrain_mode=None,
        weight_path=None,
        mode="binary",  # ["binary", "multilabel"]
        lossfn_name="dice",
        lr=1e-4,
        epochs=10,
        optim_name="adam",  # ["adam", "sgd"]
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(1, 3, 224, 224)

        self.model_name = model_name
        self.mode = mode
        self.lr = lr
        self.epochs = epochs
        self.optim_name = optim_name

        self.model = set_model(
            model_name,
            **kwargs,
        )
        if weight_path is not None:
            param = torch.load(weight_path)
            if pretrain_mode is not None:
                param = adjust_param(param, pretrain_mode)
                self.model.load_state_dict(param, strict=False)
            else:
                self.model.load_state_dict(param, strict=True)

        self.lossfn = set_lossfn(lossfn_name)
        self.optimizer = self._set_optimizers()

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
        self._shared_step(batch, phase="test")

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        out = self.model(x)
        return out

    def _set_optimizers(self):
        if self.optim_name == "adam":
            opts = {
                "optimizer": (optimizer := optim.Adam(
                    self.model.parameters(),
                    lr=self.lr,
                )),
                "lr_scheduler": optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=list(range(2,self.epochs,2)),
                    gamma=0.1,
                ),
            }

        elif self.optim_name == "sgd":
            opts = {
                "optimizer": (optimizer := optim.SGD(
                    self.model.parameters(),
                    lr=self.lr,
                )),
                "lr_scheduler": optim.lr_scheduler.PolynomialLR(
                    optimizer,
                    total_iters=self.epochs,
                    power=0.9,
                )
            }

        return opts

    def _shared_step(self, batch, phase="train"):
        frames, labels = batch[0], batch[1]
        logits = self.model(frames)

        loss = self.lossfn(logits, labels[:,0:logits.size(1),:,:])
        self.log(f"loss/{phase}", loss, on_step=True, on_epoch=True)

        metric_dict = self._calc_metric(logits, labels)
        metric_dict = {f"{k}/{phase}":v for k,v in metric_dict.items()}
        self.log_dict(metric_dict, on_step=True, on_epoch=True)

        return loss

    def _calc_metric(self, output, target):
        obj_mask, hand_mask = target[:,0:1,:,:], target[:,1:2,:,:]
        output = output[:,0:1,:,:]

        tp, fp, fn, tn = smp.metrics.get_stats(
            output,
            obj_mask.to(torch.int64),
            mode=self.mode,
            threshold=0.5,
        )

        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2 = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

        convergence = (output.sigmoid() * hand_mask).sum() / hand_mask.sum()

        return {
            "IouScore": iou,
            "F1Score": f1,
            "F2Score": f2,
            "Accuracy": acc,
            "Recall": recall,
            "Convergence": convergence,
        }
