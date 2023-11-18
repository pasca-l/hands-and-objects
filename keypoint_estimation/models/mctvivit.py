import torch
import torch.nn as nn
import transformers


class TubeletEmbeddings(nn.Module):
    def __init__(
        self, num_frames, image_size, tubelet_size, hidden_size, num_channels
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = tubelet_size
        self.num_patches = (
            (self.image_size // self.patch_size[2])
            * (self.image_size // self.patch_size[1])
            * (self.num_frames // self.patch_size[0])
        )
        self.embed_dim = hidden_size

        self.projection = nn.Conv3d(
            num_channels, hidden_size,
            kernel_size=tubelet_size, stride=tubelet_size,
        )

    def forward(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model" + \
                f"({self.image_size}*{self.image_size})."
            )

        # permute to (batch_size, num_channels, num_frames, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        # convolve to (batch_size, hidden_size, (num_patches))
        x = self.projection(pixel_values)
        # convolve to (batch_size, -1, hidden_size)
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class MCTViViTEmbeddings(nn.Module):
    def __init__(
        self,
        num_frames=16,
        image_size=224,
        tubelet_size=[2,16,16],
        cls_token_num=16,
        hidden_size=768,
        hidden_dropout_prob=0.0,
        num_channels=3,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.zeros(1, cls_token_num, hidden_size)
        )
        self.patch_embeddings = TubeletEmbeddings(
            num_frames, image_size, tubelet_size, hidden_size, num_channels
        )

        self.position_embeddings = nn.Parameter(
            torch.zeros(
                1, self.patch_embeddings.num_patches+cls_token_num, hidden_size
            )
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

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
        self.embeddings = MCTViViTEmbeddings()


class MCTViViT(nn.Module):
    def __init__(self, out_channel, with_attention):
        super().__init__()

        self.with_attention = with_attention

        config = transformers.VivitConfig(
            num_frames=out_channel,
            num_labels=1,
        )
        self.vivit = ViViT(config)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.vivit(x, output_attentions=self.with_attention)
        # print(x[0][:,0,:].shape, x.last_hidden_state.shape)
        cls_num = 16
        # logits = self.classifier(x.last_hidden_state[:,:cls_num,:])
        logits = x.last_hidden_state[:,:cls_num,:].mean(dim=-1)
        print(logits, logits.shape)

        if self.with_attention:
            # attentions: torch.Size([b, head_num, seq_size, seq_size])
            # seq_size is equivalent to tubelet_num + 1
            return logits, x.attentions

        return logits


if __name__ == "__main__":
    model = MCTViViT(16, True)
    input = torch.Tensor(1, 16, 3, 224, 224)
    model(input)
