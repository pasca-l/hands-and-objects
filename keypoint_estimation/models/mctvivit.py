import torch
import torch.nn as nn
import transformers


class TubeletEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_frames = config.num_frames
        self.image_size = config.image_size
        self.patch_size = config.tubelet_size
        self.num_patches = (
            (self.image_size // self.patch_size[2])
            * (self.image_size // self.patch_size[1])
            * (self.num_frames // self.patch_size[0])
        )
        self.embed_dim = config.hidden_size

        self.projection = nn.Conv3d(
            config.num_channels, config.hidden_size,
            kernel_size=config.tubelet_size, stride=config.tubelet_size,
        )

    def forward(self, pixel_values):
        # torch.Size([b, frame_num, ch, h, w])
        _, _, _, h, w = pixel_values.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"Input image size ({h}*{w}) doesn't match model" + \
                f"({self.image_size}*{self.image_size})."
            )

        # permute to torch.Size([b, ch, frame_num, h, w])
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        # convolve to torch.Size([b, hidden_size, (num_patches)])
        x = self.projection(pixel_values)
        # convolve to torch.Size([b, -1, hidden_size])
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class MCTViViTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.zeros(1, config.cls_token_num, config.hidden_size)
        )
        self.patch_embeddings = TubeletEmbeddings(config)

        self.position_embeddings = nn.Parameter(
            torch.zeros(
                1,
                self.patch_embeddings.num_patches + config.cls_token_num,
                config.hidden_size,
            )
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = self.cls_token.tile([batch_size, 1, 1])

        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class ViViT(transformers.VivitModel):
    def __init__(
        self, config,
    ):
        super().__init__(config)
        self.embeddings = MCTViViTEmbeddings(config)


class MCTViViT(nn.Module):
    def __init__(self, out_channel, with_attention):
        super().__init__()

        self.cls_token_num = out_channel
        self.with_attention = with_attention

        config = transformers.VivitConfig(
            num_frames=out_channel,
            num_labels=1,
            cls_token_num=16,
        )
        self.vivit = ViViT(config)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.vivit(x, output_attentions=self.with_attention)

        # last state: torch.Size([b, tubelet_num + cls_token, hidden_size])
        logits = x.last_hidden_state[:,:self.cls_token_num,:].mean(dim=-1)
        # logits = self.classifier(x.last_hidden_state[:,:cls_num,:])

        if self.with_attention:
            # attentions: torch.Size([b, head_num, seq_size, seq_size])
            # seq_size is equivalent to tubelet_num + cls_token
            return logits, x.attentions

        return logits
