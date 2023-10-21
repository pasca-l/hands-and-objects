import torch
import lightning as L
import torchmetrics


class System(L.LightningModule):
    def __init__(
        self,
        frame_num=16,
        mode="multilabel",
        inputs="all_keyframe", # ["random", "all_keyframe", "mean", ""]
    ):
        super().__init__()

        self.inputs = inputs

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

        if self.inputs == "all_keyframe":
            logits = torch.ones_like(labels)

        metrics = self._calc_metrics(logits, labels, metalabels)
        metric_dict = {f"{k}/test":v for k,v in metrics.items()}
        self.log_dict(metric_dict, on_step=True, on_epoch=True)

    def _calc_metrics(self, output, target, metalabel):
        tp, fp, tn, fn, sup = self.stats(output, target)
        stat_dict = {
            "TruePositives": tp,
            "FalsePositives": fp,
            "TrueNegatives": tn,
            "FalseNegatives": fn,
        }

        metrics = self.metrics(output, target)

        preds = output.sigmoid() > 0.5
        temp_err = (preds * metalabel).sum() / preds.sum() \
                   if preds.sum() > 0 else 0.0
        meta_metrics = {
            "AverageNearestKeyframeError": temp_err,
        }

        return stat_dict | metrics | meta_metrics
