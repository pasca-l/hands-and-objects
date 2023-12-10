import torch
from torchmetrics import Metric
import torchmetrics.functional as tmf


class TPPercentage(Metric):
    def __init__(self, task, **kwargs):
        super().__init__()
        self.task = task

        self.add_state(
            "tp_pct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds, target):
        batch_num, frame_num = preds.shape

        tp, _, _, _, _ = tmf.stat_scores(
            preds, target, task=self.task, num_labels=frame_num
        )

        self.tp_pct += tp * 100 / (batch_num * frame_num)

    def compute(self):
        return self.tp_pct


class FPPercentage(Metric):
    def __init__(self, task, **kwargs):
        super().__init__()
        self.task = task

        self.add_state(
            "fp_pct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds, target):
        batch_num, frame_num = preds.shape

        _, fp, _, _, _ = tmf.stat_scores(
            preds, target, task=self.task, num_labels=frame_num
        )

        self.fp_pct += fp * 100 / (batch_num * frame_num)

    def compute(self):
        return self.fp_pct


class TNPercentage(Metric):
    def __init__(self, task, **kwargs):
        super().__init__()
        self.task = task

        self.add_state(
            "tn_pct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds, target):
        batch_num, frame_num = preds.shape

        _, _, tn, _, _ = tmf.stat_scores(
            preds, target, task=self.task, num_labels=frame_num
        )

        self.tn_pct += tn * 100 / (batch_num * frame_num)

    def compute(self):
        return self.tn_pct


class FNPercentage(Metric):
    def __init__(self, task, **kwargs):
        super().__init__()
        self.task = task

        self.add_state(
            "fn_pct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds, target):
        batch_num, frame_num = preds.shape

        _, _, _, fn, _ = tmf.stat_scores(
            preds, target, task=self.task, num_labels=frame_num
        )

        self.fn_pct += fn * 100 / (batch_num * frame_num)

    def compute(self):
        return self.fn_pct
