import torch.nn as nn
import einops

from .vit import VisionTransformerWithoutHead


# Patch Convoluting ViViT
class PCViViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        num_in_frames=16,
        num_out_frames=16,
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
    ):
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.with_attention = with_attention

        self.ptoken_temp_size = num_in_frames // patch_size[0]
        self.ptoken_space_size = image_size // patch_size[1]

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

        self.cnn = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size * 2,
                kernel_size=(3, 5, 5),
                stride=2,
            ),
            nn.BatchNorm3d(hidden_size * 2),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=hidden_size * 2,
                out_channels=hidden_size,
                kernel_size=(2, 3, 3),
                stride=2,
            ),
            nn.BatchNorm3d(hidden_size),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(intermediate_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_out_frames),
        )

    def forward(self, x):
        x, attn = self.vivit(x)

        cls_token = x[:,:self.num_cls_tokens,:]
        patch_token = x[:,self.num_cls_tokens:,:]

        x = einops.rearrange(
            patch_token, "b (f h w) c -> b c f h w",
            f=self.ptoken_temp_size,
            h=self.ptoken_space_size,
        )
        x = self.cnn(x)
        logits = self.head(x)

        if self.with_attention:
            # attentions: torch.Size([b, head_num, seq_size, seq_size]) x blocks
            # seq_size is equivalent to patch_num + 1
            return logits, attn

        return logits
