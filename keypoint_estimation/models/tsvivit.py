import torch
import torch.nn as nn

from .vit import VisionTransformerWithoutHead


# Token Scoring ViViT
# inspired form https://arxiv.org/pdf/2111.11591.pdf (scorer network)
class TSViViT(nn.Module):
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
    ):
        super().__init__()
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

        self.patch_n = (
            (image_size // patch_size[2])
            * (image_size // patch_size[1])
        )
        self.patch_t = num_frames // patch_size[0]

        self.local_patch = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Flatten(),
            nn.Linear(num_frames // 2, num_frames),
        )

    def forward(self, x):
        x, attn = self.vivit(x)

        cls_token = x[:,:1,:]
        patch_token = x[:,1:,:]

        # average patch tokens to get temporal tokens
        temp_token = patch_token.reshape(
            x.size(0), self.patch_t, self.patch_n, -1
        )
        x = torch.mean(temp_token, dim=2)

        # scores of temporal tokens
        local_x = self.local_patch(x)
        global_x = torch.mean(local_x, dim=1, keepdim=True)

        x = torch.cat([local_x, global_x.expand(local_x.shape)], dim=-1)
        logits = self.score_head(x)

        # min-max normalization of scores
        # logits = (logits - logits.min(axis=-1, keepdim=True).values) / (logits.max(axis=-1, keepdim=True).values - logits.min(axis=-1, keepdim=True).values + 1e-5)

        if self.with_attention:
            # attentions: torch.Size([b, head_num, seq_size, seq_size]) x blocks
            # seq_size is equivalent to patch_num + 1
            return logits, attn

        return logits
