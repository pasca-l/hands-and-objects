import torchmetrics

from .stats import (
    TPPercentage,
    FPPercentage,
    TNPercentage,
    FNPercentage,
)
from .keyframe_error import (
    AverageGlobalNearestKeyframeError,
    AverageLocalNearestKeyframeError,
    AverageKeyframeNumError,
)


def set_metrics(task="multilabel", num_labels=16, thresholds=10):
    metrics = torchmetrics.MetricCollection([
        TPPercentage(task=task),
        FPPercentage(task=task),
        TNPercentage(task=task),
        FNPercentage(task=task),
        torchmetrics.Accuracy(task=task, num_labels=num_labels),
        torchmetrics.Precision(task=task, num_labels=num_labels),
        torchmetrics.Recall(task=task, num_labels=num_labels),
        torchmetrics.F1Score(task=task, num_labels=num_labels),
        torchmetrics.AveragePrecision(
            task=task, num_labels=num_labels, thresholds=thresholds
        ),
        torchmetrics.AUROC(
            task=task, num_labels=num_labels, thresholds=thresholds
        ),
        AverageKeyframeNumError(),
    ])

    return metrics


def set_meta_metrics():
    meta_metrics = torchmetrics.MetricCollection([
        AverageGlobalNearestKeyframeError(),
        AverageLocalNearestKeyframeError(),
    ])

    return meta_metrics
