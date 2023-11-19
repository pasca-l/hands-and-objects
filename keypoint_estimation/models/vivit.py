import torch.nn as nn
import transformers


class ViViT(nn.Module):
    def __init__(
        self,
        out_channel,
        with_attention,
    ):
        super().__init__()

        self.with_attention = with_attention

        config = transformers.VivitConfig(
            num_frames=out_channel,
            num_labels=out_channel,
        )
        num_patches = (
            (config.image_size // config.tubelet_size[2])
            * (config.image_size // config.tubelet_size[1])
            * (config.num_frames // config.tubelet_size[0])
        )
        self.vivit = transformers.VivitModel(config)

        self.norm = nn.LayerNorm(config.hidden_size)
        self.cls_head = nn.Linear(config.hidden_size, config.num_labels)
        self.patch_head = nn.Linear(num_patches, config.num_labels)

    def forward(self, x):
        output = self.vivit(x, output_attentions=self.with_attention)
        x, att = output.last_hidden_state, output.attentions

        cls_token = x[:,:1,:]
        patch_token = x[:,1:,:]

        # normal logits from class token
        cls_token = self.norm(cls_token)
        cls_logits = self.cls_head(cls_token.squeeze(1))

        # use patch tokens for logits
        # average across hidden dimension size
        patch_token = self.norm(patch_token)
        patch_token = patch_token.mean(dim=-1)
        patch_logits = self.patch_head(patch_token)

        logits = patch_logits

        if self.with_attention:
            # attentions: torch.Size([b, head_num, seq_size, seq_size]) x blocks
            # seq_size is equivalent to tubelet_num + 1
            return logits, att

        return logits
