import torch
from torchmetrics import Metric


class AverageNearestKeyframeError(Metric):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

        self.add_state(
            "nearest_err", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, logits, metalabel):
        # assume input as logits
        preds = logits.sigmoid() > self.threshold
        err = (preds * metalabel).sum() / preds.sum() \
                if preds.sum() > 0 else 0.0

        self.nearest_err += err

    def compute(self):
        return self.nearest_err


class AverageKeyframeNumError(Metric):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

        self.add_state(
            "num_err", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, logits, target):
        # assume input as logits
        preds = logits.sigmoid() > self.threshold
        err = torch.abs(preds.sum() - target.sum())

        self.num_err += err

    def compute(self):
        return self.num_err
