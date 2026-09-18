"""Dependency-light binary metrics for dropout prediction."""
from __future__ import annotations

import numpy as np


def _binary_inputs(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be non-empty and have equal shape")
    if not np.all((labels == 0) | (labels == 1)) or not np.isfinite(scores).all():
        raise ValueError("labels must be binary and scores finite")
    if labels.min() == labels.max():
        raise ValueError("Both classes are required for AUC/AUPRC")
    return labels, scores


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = _binary_inputs(labels, scores)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = labels.size - positive_count
    positive_rank_sum = ranks[positives].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = _binary_inputs(labels, scores)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    cumulative_positives = np.cumsum(sorted_labels)
    group_ends = np.r_[
        np.flatnonzero(np.diff(sorted_scores) != 0), sorted_scores.size - 1
    ]
    true_positives = cumulative_positives[group_ends]
    predicted_positives = group_ends + 1
    precision = true_positives / predicted_positives
    recall = true_positives / cumulative_positives[-1]
    recall_increase = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increase * precision))


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels, probabilities = _binary_inputs(labels, probabilities)
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    predictions = probabilities >= threshold
    positives = labels == 1
    true_positive = int(np.count_nonzero(predictions & positives))
    false_positive = int(np.count_nonzero(predictions & ~positives))
    false_negative = int(np.count_nonzero(~predictions & positives))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
    return {
        "auc": roc_auc(labels, probabilities),
        "auprc": average_precision(labels, probabilities),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }
