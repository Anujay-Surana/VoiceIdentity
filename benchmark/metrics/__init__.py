"""Evaluation metrics for speaker diarization and identification."""

from benchmark.metrics.diarization import (
    DiarizationErrorRate,
    compute_der,
    compute_jer,
    compute_coverage,
    compute_purity,
)
from benchmark.metrics.identification import (
    compute_identification_accuracy,
    compute_eer,
    compute_confusion_matrix,
)

__all__ = [
    "DiarizationErrorRate",
    "compute_der",
    "compute_jer",
    "compute_coverage",
    "compute_purity",
    "compute_identification_accuracy",
    "compute_eer",
    "compute_confusion_matrix",
]
