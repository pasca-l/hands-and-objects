import torchmetrics

from .stats import (
    TPPercentage,
    FPPercentage,
    TNPercentage,
    FNPercentage,
    MeanAveragePrecision,
)
from .keyframe_error import (
    AverageGlobalNearestKeyframeError,
    AverageLocalNearestKeyframeError,
    AverageKeyframeNumError,
)


def set_metrics(mode="multilabel", frame_num=16):
    metrics = torchmetrics.MetricCollection([
        TPPercentage(task=mode),
        FPPercentage(task=mode),
        TNPercentage(task=mode),
        FNPercentage(task=mode),
        torchmetrics.Accuracy(task=mode, num_labels=frame_num),
        torchmetrics.Precision(task=mode, num_labels=frame_num),
        torchmetrics.Recall(task=mode, num_labels=frame_num),
        torchmetrics.F1Score(task=mode, num_labels=frame_num),
        MeanAveragePrecision(task=mode),
        AverageKeyframeNumError(),
    ])

    return metrics


def set_meta_metrics():
    meta_metrics = torchmetrics.MetricCollection([
        AverageGlobalNearestKeyframeError(),
        AverageLocalNearestKeyframeError(),
    ])

    return meta_metrics
