import torch
from torchmetrics import Metric


class AverageGlobalNearestKeyframeError(Metric):
    def __init__(self, threshold=0.5, in_sec=True, fps=30):
        super().__init__()
        self.threshold = threshold
        self.in_sec = in_sec
        self.fps = fps

        self.add_state(
            "nearest_err", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, logits, target, metalabel):
        # assume input as logits
        preds = logits.sigmoid() > self.threshold
        err = ((preds * metalabel).sum(dim=1) / preds.sum(dim=1)).nan_to_num()
        err = err.mean()

        self.nearest_err += (err / self.fps) if self.in_sec else err

    def compute(self):
        return self.nearest_err


class AverageLocalNearestKeyframeError(Metric):
    def __init__(self, threshold=0.5, in_sec=True, fps=30):
        super().__init__()
        self.threshold = threshold
        self.in_sec = in_sec
        self.fps = fps

        self.add_state(
            "nearest_err", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, logits, target, metalabel):
        # assume input as logits
        preds = logits.sigmoid() > self.threshold
        err = ((preds * metalabel).sum(dim=1) / preds.sum(dim=1)).nan_to_num()

        # exclude error with labels without keyframe
        inclusion = target.sum(dim=1).clamp(max=1)
        err = ((err * inclusion).sum() / inclusion.sum()).nan_to_num()

        self.nearest_err += (err / self.fps) if self.in_sec else err

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
