import torch
import torch.nn as nn
import torch.optim as optim
import einops
import lightning as L
import transformers
import segmentation_models_pytorch as smp


class System(L.LightningModule):
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

        self.model = TransUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channel=in_channels,
            out_channel=out_channels,
        )
        self.lossfn = self._set_lossfn()
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
            "IouScore": iou,
            "F1Score": f1,
            "F2Score": f2,
            "Accuracy": acc,
            "Recall": recall,
            "Convergence": convergence,
        }


class TransUNet(nn.Module):
    def __init__(
        self,
        encoder_name,
        encoder_weights,
        in_channel,
        out_channel,
    ):
        super().__init__()

        # disassemble modules to add ViT encoder
        unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channel,
            classes=out_channel,
        )

        self.encoder_conv1 = unet.encoder.conv1
        self.encoder_bn1 = unet.encoder.bn1
        self.encoder_relu = unet.encoder.relu
        self.encoder_maxpool = unet.encoder.maxpool
        self.encoder_layer1 = unet.encoder.layer1
        self.encoder_layer2 = unet.encoder.layer2
        self.encoder_layer3 = unet.encoder.layer3
        self.encoder_layer4 = unet.encoder.layer4

        self.decoder = unet.decoder
        self.segmentation_head = unet.segmentation_head

        config = transformers.ViTConfig(
            image_size=7,
            patch_size=1,
            num_channels=2048
        )
        self.vit = transformers.ViTModel(config)
        self.project_back = nn.Linear(in_features=768, out_features=2048)
        self.vit_conv = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # input: [batch, 3, 224, 224]
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        feat0 = self.encoder_relu(x)
        x = self.encoder_maxpool(feat0)
        feat1 = self.encoder_layer1(x)
        feat2 = self.encoder_layer2(feat1)
        feat3 = self.encoder_layer3(feat2)
        feat4 = self.encoder_layer4(feat3)

        # after ResNet: [batch, 2048, 7, 7]
        y = self.vit(feat4).last_hidden_state
        # after ViT: [batch, patch_num + cls_head, hidden_size]
        cls_head, x = y[:,0,:], y[:,1:,:]

        x = self.project_back(x)
        x = einops.rearrange(x, "b (x y) c -> b c x y", x=7, y=7)
        x = self.vit_conv(x)

        x = self.decoder(x, feat0, feat1, feat2, feat3, feat4)
        x = self.segmentation_head(x)

        return x
