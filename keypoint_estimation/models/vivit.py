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
        self.vivit = transformers.VivitForVideoClassification(config)

    def forward(self, x):
        x = self.vivit(x, output_attentions=self.with_attention)

        if self.with_attention:
            return x.logits, x.attentions

        return x.logits
