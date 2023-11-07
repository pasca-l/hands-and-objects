import torchmetrics

from stats import (
    TPPercentage,
    FPPercentage,
    TNPercentage,
    FNPercentage,
)
from keyframe_error import (
    AverageNearestKeyframeError,
    AverageKeyframeNumError,
)


def set_metrics(mode="multilabel", frame_num=16):
    metrics = torchmetrics.MetricCollection([
        TPPercentage(task=mode, num_labels=frame_num),
        FPPercentage(task=mode, num_labels=frame_num),
        TNPercentage(task=mode, num_labels=frame_num),
        FNPercentage(task=mode, num_labels=frame_num),
        torchmetrics.Accuracy(task=mode, num_labels=frame_num),
        torchmetrics.Precision(task=mode, num_labels=frame_num),
        torchmetrics.Recall(task=mode, num_labels=frame_num),
        torchmetrics.F1Score(task=mode, num_labels=frame_num),
        AverageKeyframeNumError(),
    ])

    return metrics


def set_meta_metrics():
    meta_metrics = torchmetrics.MetricCollection([
        AverageNearestKeyframeError(),
    ])

    return meta_metrics
