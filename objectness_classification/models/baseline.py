import numpy as np
import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(
        self,
        baseline="random",
        num_out_frames=16,
        seed=42,
    ):
        super().__init__()

        self.dummy_param = nn.parameter.Parameter(torch.Tensor(0))

        self.baseline = baseline
        self.out_channel = num_out_frames
        self.seed = seed

    def forward(self, x):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        logits = torch.randint(
            2, (x.shape[0], self.out_channel),
            device=x.get_device(),
        )

        return logits
