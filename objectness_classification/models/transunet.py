import torch.nn as nn
import einops
import segmentation_models_pytorch as smp

from .vit import VisionTransformerWithoutHead


class TransUNet(nn.Module):
    def __init__(
        self,
        encoder_name="resnet101",
        encoder_weights="imagenet",
        unet_in_channels=3,
        image_size=7,
        num_in_frames=1,
        num_out_frames=1,
        patch_size=[1, 1, 1],
        in_channels=2048,
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
    ):
        super().__init__()

        self.with_attention = with_attention

        # disassemble modules to add ViT encoder
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

        self.decoder = unet.decoder
        self.segmentation_head = unet.segmentation_head

        self.vit = VisionTransformerWithoutHead(
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
        y, attn = self.vit(feat4.unsqueeze(1))
        # after ViT: [batch, patch_num + cls_head, hidden_size]
        cls_head, x = y[:,0,:], y[:,1:,:]

        x = self.project_back(x)
        x = einops.rearrange(x, "b (x y) c -> b c x y", x=7, y=7)
        x = self.vit_conv(x)

        x = self.decoder(x, feat0, feat1, feat2, feat3, feat4)
        x = self.segmentation_head(x)

        if self.with_attention:
            return x, attn

        return x
