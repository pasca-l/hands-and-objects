import torch.nn as nn
import transformers


class ViViT(nn.Module):
    def __init__(
        self,
        out_channel,
    ):
        super().__init__()

        config = transformers.VivitConfig(
            num_frames=out_channel,
        )
        self.vivit = transformers.VivitModel(config)
        self.fc_norm = nn.LayerNorm((768,))
        self.classifier = nn.Linear(in_features=768, out_features=out_channel)

    def forward(self, x):
        x = self.vivit(x).last_hidden_state
        x = self.fc_norm(x.mean(1))
        x = self.classifier(x)

        return x
