import torch
from torchmetrics import Metric
import torchmetrics.functional as tmf


class TPPercentage(Metric):
    def __init__(self, task, num_labels):
        super().__init__()
        self.task = task
        self.num_labels = num_labels

        self.add_state(
            "tp_pct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds, target):
        batch_num = preds.shape[0]
        frame_num = self.num_labels

        tp, _, _, _, _ = tmf.stat_scores(
            preds, target, task=self.task, num_labels=self.num_labels
        )

        self.tp_pct += tp * 100 / (batch_num * frame_num)

    def compute(self):
        return self.tp_pct


class FPPercentage(Metric):
    def __init__(self, task, num_labels):
        super().__init__()
        self.task = task
        self.num_labels = num_labels

        self.add_state(
            "fp_pct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds, target):
        batch_num = preds.shape[0]
        frame_num = self.num_labels

        _, fp, _, _, _ = tmf.stat_scores(
            preds, target, task=self.task, num_labels=self.num_labels
        )

        self.fp_pct += fp * 100 / (batch_num * frame_num)

    def compute(self):
        return self.fp_pct


class TNPercentage(Metric):
    def __init__(self, task, num_labels):
        super().__init__()
        self.task = task
        self.num_labels = num_labels

        self.add_state(
            "tn_pct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds, target):
        batch_num = preds.shape[0]
        frame_num = self.num_labels

        _, _, tn, _, _ = tmf.stat_scores(
            preds, target, task=self.task, num_labels=self.num_labels
        )

        self.tn_pct += tn * 100 / (batch_num * frame_num)

    def compute(self):
        return self.tn_pct


class FNPercentage(Metric):
    def __init__(self, task, num_labels):
        super().__init__()
        self.task = task
        self.num_labels = num_labels

        self.add_state(
            "fn_pct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds, target):
        batch_num = preds.shape[0]
        frame_num = self.num_labels

        _, _, _, fn, _ = tmf.stat_scores(
            preds, target, task=self.task, num_labels=self.num_labels
        )

        self.fn_pct += fn * 100 / (batch_num * frame_num)

    def compute(self):
        return self.fn_pct
