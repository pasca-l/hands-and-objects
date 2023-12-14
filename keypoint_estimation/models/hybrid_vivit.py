import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from .vit import VisionTransformerWithoutHead


# inspired from hybrid architecture of ViT
# originally from https://arxiv.org/pdf/2010.11929.pdf
class HybridViViT(nn.Module):
    def __init__(
        self,
        encoder_name="resnet18",
        encoder_weights="imagenet",
        unet_in_channels=3,
        image_size=14,
        num_in_frames=8,
        num_out_frames=1,
        patch_size=[1, 1, 1],
        in_channels=1024,
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
        self.num_cls_tokens = num_cls_tokens
        self.with_attention = with_attention

        unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=unet_in_channels,
            classes=num_out_frames,
        )

        self.encoder_conv1 = unet.encoder.conv1
        self.encoder_bn1 = unet.encoder.bn1
        self.encoder_relu = unet.encoder.relu
        self.encoder_maxpool = unet.encoder.maxpool
        self.encoder_layer1 = unet.encoder.layer1
        self.encoder_layer2 = unet.encoder.layer2
        self.encoder_layer3 = unet.encoder.layer3
        self.encoder_layer4 = unet.encoder.layer4

        self.upsample = nn.Upsample(
            scale_factor=2,
        )

        self.conv1d = nn.Conv3d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
        )

        self.vivit = VisionTransformerWithoutHead(
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

        self.cls_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_out_frames),
        )

    def image_conv(self, x):
        # input: torch.Size([b, 3, 224, 224])
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        feat0 = self.encoder_relu(x)
        x = self.encoder_maxpool(feat0)
        feat1 = self.encoder_layer1(x)
        feat2 = self.encoder_layer2(feat1)
        feat3 = self.encoder_layer3(feat2)
        feat4 = self.encoder_layer4(feat3)

        # torch.Size([b, 512, 7, 7]) to torch.Size([b, 512, 14, 14])
        x = self.upsample(feat4)
        return x

    def forward(self, x):
        # get spatial features from 2d cnn
        features = [
            self.image_conv(x[:,i,:,:,:])
            for i in range(x.size(1))
        ]
        x = torch.stack(features, dim=1)

        # get temporal features from 1d cnn
        # torch.Size([b, f, 512, 14, 14]) to torch.Size([b, 512, f, 14, 14])
        x = x.permute(0,2,1,3,4)
        x = self.conv1d(x)
        # permute back
        x = x.permute(0,2,1,3,4)

        x, attn = self.vivit(x)

        cls_token = x[:,:self.num_cls_tokens,:]
        patch_token = x[:,self.num_cls_tokens:,:]

        logits = self.cls_head(cls_token).squeeze(1)

        if self.with_attention:
            # attentions: torch.Size([b, head_num, seq_size, seq_size]) x blocks
            # seq_size is equivalent to patch_num + 1
            return logits, attn

        return logits
