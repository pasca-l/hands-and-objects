import torch.nn as nn
import einops

from .vit import VisionTransformerWithoutHead


class SETR(nn.Module):
    def __init__(
        self,
        decoder_name="pup",  # ["naive", "pup"]
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
    ):
        super().__init__()
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

        if decoder_name == "naive":
            self.decoder = NaiveUpsample(
                out_channels=num_out_frames,
                image_size=image_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
            )

        elif decoder_name == "pup":
            self.decoder = ProgressiveUpsample(
                out_channels=num_out_frames,
                image_size=image_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
            )

    def forward(self, x):
        x = x.unsqueeze(1)
        x, attn = self.encoder(x)

        patch_token = x[:,self.num_cls_tokens:,:]

        logits = self.decoder(patch_token)

        if self.with_attention:
            return logits, attn

        return logits


class NaiveUpsample(nn.Module):
    def __init__(
        self,
        out_channels,
        image_size,
        patch_size,
        hidden_size,
    ):
        super().__init__()

        self.patch_token_size = image_size // patch_size[1]

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=1,
            ),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_size,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )
        self.upsample = nn.Upsample(scale_factor=patch_size[1], mode="bilinear")

    def forward(self, x):
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=self.patch_token_size)
        x = self.head(x)
        x = self.upsample(x)

        return x


class ProgressiveUpsample(nn.Module):
    def __init__(
        self,
        out_channels,
        image_size,
        patch_size,
        hidden_size,
    ):
        super().__init__()

        self.patch_token_size = image_size // patch_size[1]

        self.conv_first = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size // 4,
            kernel_size=1,
        )
        self.conv_inter = nn.Conv2d(
            in_channels=hidden_size // 4,
            out_channels=hidden_size // 4,
            kernel_size=1,
        )
        self.conv_last = nn.Conv2d(
            in_channels=hidden_size // 4,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=self.patch_token_size)

        x = self.upsample(self.conv_first(x))
        x = self.upsample(self.conv_inter(x))
        x = self.upsample(self.conv_inter(x))
        x = self.upsample(self.conv_inter(x))
        x = self.conv_last(x)

        return x
