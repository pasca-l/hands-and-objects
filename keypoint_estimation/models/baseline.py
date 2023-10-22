import sys
import numpy as np
import torch
import lightning as L
import torchmetrics


class System(L.LightningModule):
    def __init__(
        self,
        frame_num=16,
        mode="multilabel",
        inputs="random", # ["random", "choice"]
        seed=42,
        choice_num=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.frame_num = frame_num
        self.inputs = inputs
        self.choice_num = choice_num

        self.stats = torchmetrics.StatScores(task=mode, num_labels=frame_num)
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task=mode, num_labels=frame_num),
            torchmetrics.Precision(task=mode, num_labels=frame_num),
            torchmetrics.Recall(task=mode, num_labels=frame_num),
            torchmetrics.F1Score(task=mode, num_labels=frame_num),
        ])

        self.hparams.update({"model": "Baseline"})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        _, labels, metalabels = batch[0], batch[1], batch[2]
        labels = labels.float()

        if self.inputs == "random":
            logits = torch.randint(
                2, labels.shape, device=labels.get_device()
            )

        elif self.inputs == "choice":
            logits = torch.stack([torch.tensor(
                np.where(np.isin(
                    np.arange(self.frame_num),
                    np.random.choice(
                        self.frame_num, self.choice_num, replace=False
                    )
                ), 1, 0),
                device=labels.get_device(),
            ) for _ in range(labels.shape[0])])

        metrics = self._calc_metrics(logits, labels, metalabels)
        metric_dict = {f"{k}/test":v for k,v in metrics.items()}
        self.log_dict(metric_dict, on_step=True, on_epoch=True)

    def _calc_metrics(self, output, target, metalabel):
        # stats (tp, fp, tn, fn) is converted to percentage,
        # as value increases, with greater batch size and input frame number
        batch_num = output.shape[0]
        tp, fp, tn, fn, sup = self.stats(output, target)
        stat_dict = {
            "TruePositives": tp * 100 / (batch_num * self.frame_num),
            "FalsePositives": fp * 100 / (batch_num * self.frame_num),
            "TrueNegatives": tn * 100 / (batch_num * self.frame_num),
            "FalseNegatives": fn * 100 / (batch_num * self.frame_num),
        }

        # preds outside of [0,1] will be considered as logits,
        # and sigmoid() is auto applied
        metrics = self.metrics(output, target)

        # metalabel contains the nearest temporal error,
        # so relevant values are summed
        preds = output.sigmoid() > 0.5
        temp_err = (preds * metalabel).sum() / preds.sum() \
                   if preds.sum() > 0 else 0.0
        meta_metrics = {
            "AverageNearestKeyframeError": temp_err,
        }

        return stat_dict | metrics | meta_metrics
