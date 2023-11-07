import numpy as np
import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(
        self,
        baseline="random", # ["random", "choice"]
        out_channel=16,
        seed=42,
        choice_num=1,
    ):
        super().__init__()

        self.dummy_param = nn.parameter.Parameter(torch.Tensor(0))

        self.baseline = baseline
        self.out_channel = out_channel
        self.seed = seed
        self.choice_num = choice_num

    def forward(self, x):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.baseline == "random":
            logits = torch.randint(
                2, (x.shape[0], self.out_channel),
                device=x.get_device(),
            )

        elif self.baseline == "choice":
            logits = torch.stack([torch.tensor(
                np.where(np.isin(
                    np.arange(self.out_channel),
                    np.random.choice(
                        self.out_channel, self.choice_num, replace=False
                    )
                ), 1, 0),
                device=x.get_device(),
            ) for _ in range(x.shape[0])])

        return logits
