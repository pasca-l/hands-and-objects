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


def set_metrics(**kwargs):
    metrics = torchmetrics.MetricCollection([
        TPPercentage(**kwargs),
        FPPercentage(**kwargs),
        TNPercentage(**kwargs),
        FNPercentage(**kwargs),
        torchmetrics.Accuracy(**kwargs),
        torchmetrics.Precision(**kwargs),
        torchmetrics.Recall(**kwargs),
        torchmetrics.F1Score(**kwargs),
        torchmetrics.AveragePrecision(**kwargs),
        torchmetrics.AUROC(**kwargs),
        AverageKeyframeNumError(),
    ])

    return metrics


def set_meta_metrics():
    meta_metrics = torchmetrics.MetricCollection([
        AverageGlobalNearestKeyframeError(),
        AverageLocalNearestKeyframeError(),
    ])

    return meta_metrics
