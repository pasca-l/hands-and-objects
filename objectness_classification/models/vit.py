import torch
import torch.nn as nn
import torch.nn.functional as nnf


class VisionTransformerWithoutHead(nn.Module):
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
    ):
        super().__init__()
        self.image_size = image_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_cls_tokens = num_cls_tokens
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.qkv_bias = qkv_bias
        self.intermediate_size = intermediate_size
        self.with_attn_weights = with_attn_weights

        self.embedding = VisionTransformerEmbeddings(
            image_size=image_size,
            num_frames=num_frames,
            patch_size=patch_size,
            in_channels=in_channels,
            num_cls_tokens=num_cls_tokens,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
        )

        self.encoder = nn.ModuleList([
            VisionTransformerEncoderBlock(
                num_heads=num_heads,
                hidden_size=hidden_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_dropout_prob=attention_dropout_prob,
                qkv_bias=qkv_bias,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        attn_weights = []

        x = self.embedding(x)

        for block in self.encoder:
            x, attn = block(x)
            attn_weights.append(attn)

        if self.with_attn_weights:
            return x, attn_weights

        return x


class VisionTransformerEmbeddings(nn.Module):
    def __init__(
        self,
        image_size,
        num_frames,
        patch_size,
        in_channels,
        num_cls_tokens,
        hidden_size,
        hidden_dropout_prob,
    ):
        super().__init__()

        num_patches = (
            (image_size // patch_size[2])
            * (image_size // patch_size[1])
            * (num_frames // patch_size[0])
        )

        # patch embedding
        self.projection = nn.Conv3d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, num_cls_tokens, hidden_size)
        )

        # positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_cls_tokens + num_patches, hidden_size)
        )

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        # torch.Size([b, frame_num, ch, h, w])
        # permute to torch.Size([b, ch, frame_num, h, w])
        x = x.permute(0, 2, 1, 3, 4)

        # convolve to torch.Size([b, hidden_size, num_patches])
        x = self.projection(x).flatten(start_dim=2)
        # transpose to torch.Size([b, num_patches, hidden_size])
        x = x.transpose(1, 2)

        # add cls tokens (concat to head)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.tile([batch_size, 1, 1])
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional embedding
        x = x + self.pos_embed

        # torch.Size([b, num_cls_tokens + num_patches, hidden_size])
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_size,
        hidden_dropout_prob,
        attention_dropout_prob,
        qkv_bias,
    ):
        super().__init__()

        self.num_head = num_heads
        self.head_size = hidden_size // num_heads
        self.sqrt_dh = self.head_size ** 0.5
        self.hidden_size = hidden_size

        self.w_query = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.w_key = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.w_value = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        self.dropout = nn.Dropout(attention_dropout_prob)
        self.w_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(hidden_dropout_prob),
        )

    def forward(self, x):
        # torch.Size([b, num_patches, hidden_size])
        batch_size, num_patches, _ = x.size()

        q = self.w_query(x)
        k = self.w_key(x)
        v = self.w_value(x)

        def split_for_head(x):
            # divide vector for each head
            # to torch.Size([b, num_patches, num_head, head_size])
            x = x.view(batch_size, num_patches, self.num_head, self.head_size)
            # transpose to torch.Size([b, num_head, num_patches, head_size])
            x = x.transpose(1, 2)
            return x

        q = split_for_head(q)
        k = split_for_head(k)
        v = split_for_head(v)

        k_T = k.transpose(2, 3)
        dot_prod = torch.matmul(q, k_T) / self.sqrt_dh
        # attention is torch.Size([b, num_head, num_patches, num_patches])
        attn = nnf.softmax(dot_prod, dim=-1)

        attn = self.dropout(attn)

        # output to torch.Size([b, num_head, num_patches, head_size])
        out = torch.matmul(attn, v)
        # reshape to torch.Size([b, num_patches, hidden_size])
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, num_patches, self.hidden_size)

        out = self.w_output(out)

        return out, attn


class VisionTransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_size,
        hidden_dropout_prob,
        attention_dropout_prob,
        qkv_bias,
        intermediate_size,
    ):
        super().__init__()

        self.ln_before = nn.LayerNorm(hidden_size)
        self.ln_after = nn.LayerNorm(hidden_size)
        self.mhsa = MultiHeadSelfAttention(
            num_heads=num_heads,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_dropout_prob=attention_dropout_prob,
            qkv_bias=qkv_bias,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(hidden_dropout_prob),
        )

    def forward(self, x):
        out, attn = self.mhsa(self.ln_before(x))
        out = out + x
        out = self.mlp(self.ln_after(out)) + out
        return out, attn
