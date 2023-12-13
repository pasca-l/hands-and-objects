import torch.nn as nn
import torch.nn.functional as nnf
import einops

from .vit import VisionTransformerWithoutHead


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
