import torch
from torchmetrics import Metric


class AverageNearestKeyframeError(Metric):
    def __init__(self, threshold=0.5, in_sec=True, fps=30):
        super().__init__()
        self.threshold = threshold
        self.in_sec = in_sec
        self.fps = fps

        self.add_state(
            "nearest_err", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, logits, metalabel):
        batch_num = logits.shape[0]

        # assume input as logits
        preds = logits.sigmoid() > self.threshold
        err = (preds * metalabel).sum() / preds.sum() \
                if preds.sum() > 0 else 0.0

        if self.in_sec:
            self.nearest_err += err / self.fps
        else:
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
        batch_num = logits.shape[0]

        # assume input as logits
        preds = logits.sigmoid() > self.threshold
        err = torch.abs(preds.sum() - target.sum())

        self.num_err += err / batch_num

    def compute(self):
        return self.num_err
