import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import einops
import lightning as L
import segmentation_models_pytorch as smp

from vit import VisionTransformerWithoutHead


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

        self.model = Segmenter(
            num_out_frames=out_channels,
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


class Segmenter(nn.Module):
    def __init__(
        self,
        image_size=224,
        num_in_frames=1,
        num_out_frames=1,
        patch_size=[1, 16, 16],
        in_channels=3,
        num_cls_tokens=1,
        hidden_size=768,
        hidden_dropout_prob=0.0,
        num_blocks=12,
        num_heads=12,
        attention_dropout_prob=0.0,
        qkv_bias=False,
        intermediate_size=3072,
        with_attn_weights=True,
        with_attention=False,
        logit_mode="default",  # ["default", "p_hidden", "p_tnum"]
    ):
        super().__init__()
        self.logit_mode = logit_mode
        self.image_size = image_size
        self.num_cls_tokens = num_cls_tokens
        self.with_attention = with_attention

        self.encoder = VisionTransformerWithoutHead(
            image_size=image_size,
            num_frames=num_in_frames,
            patch_size=patch_size,
            in_channels=in_channels,
            num_cls_tokens=num_cls_tokens,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            num_blocks=num_blocks,
            num_heads=num_heads,
            attention_dropout_prob=attention_dropout_prob,
            qkv_bias=qkv_bias,
            intermediate_size=intermediate_size,
            with_attn_weights=with_attn_weights,
        )

        self.decoder = LinearDecoder(
            out_channels=num_out_frames,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x, attn = self.encoder(x)

        patch_token = x[:,self.num_cls_tokens:,:]

        masks = self.decoder(patch_token)
        masks = nnf.interpolate(
            masks,
            size=(self.image_size, self.image_size),
            mode="bilinear"
        )

        if self.with_attention:
            return masks, attn

        return masks


class LinearDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        image_size,
        patch_size,
        hidden_size,
    ):
        super().__init__()

        self.head = nn.Linear(hidden_size, out_channels)
        self.patch_token_size = image_size // patch_size[1]

    def forward(self, x):
        x = self.head(x)
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=self.patch_token_size)

        return x
