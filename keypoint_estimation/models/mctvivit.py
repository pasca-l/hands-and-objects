import torch.nn as nn

from vit import VisionTransformerWithoutHead


# Multi-Class Token ViViT
# inspired from https://arxiv.org/pdf/2203.02891.pdf (using multiple cls token)
class MCTViViT(nn.Module):
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

    def forward(self, x):
        x, attn = self.vivit(x)

        cls_token = x[:,:self.num_cls_tokens,:]

        # multi-class token (taking simple mean across hidden_dim)
        
        logits = cls_token.mean(dim=-1)

        if self.with_attention:
            # attentions: torch.Size([b, head_num, seq_size, seq_size]) x blocks
            # seq_size is equivalent to patch_num + 1
            return logits, attn

        return logits
