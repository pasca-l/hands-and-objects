import torch.nn as nn

from .vit import VisionTransformerWithoutHead


class ViViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        num_frames=16,
        patch_size=[2, 16, 16],
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
        self.num_cls_tokens = num_cls_tokens
        self.with_attention = with_attention

        self.vivit = VisionTransformerWithoutHead(
            image_size=image_size,
            num_frames=num_frames,
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
            nn.Linear(hidden_size, num_frames),
        )

        num_patches = (
            (image_size // patch_size[2])
            * (image_size // patch_size[1])
            * (num_frames // patch_size[0])
        )
        self.patch_head = nn.Sequential(
            nn.LayerNorm(num_patches),
            nn.Linear(num_patches, num_frames),
        )

    def forward(self, x):
        x, attn = self.vivit(x)

        cls_token = x[:,:self.num_cls_tokens,:]
        patch_token = x[:,self.num_cls_tokens:,:]

        logits = {
            # common logits from class token
            "default": self.cls_head(cls_token).squeeze(1),

            # average across patch token hidden dimension size
            "p_hidden": self.patch_head(patch_token.mean(dim=-1)),

            # average across patch token number dimension size
            "p_tnum": self.cls_head(patch_token.mean(dim=1)),
        }[self.logit_mode]

        if self.with_attention:
            # attentions: torch.Size([b, head_num, seq_size, seq_size]) x blocks
            # seq_size is equivalent to patch_num + 1
            return logits, attn

        return logits
