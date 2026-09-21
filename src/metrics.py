# Các metric phân loại nhị phân dùng chung cho validation và test.
from __future__ import annotations

import numpy as np


# Mục đích: Chuẩn hóa và kiểm tra label/score trước khi tính metric.
# Đầu vào: labels nhị phân và scores hoặc probabilities cùng số phần tử.
# Đầu ra: Hai vector NumPy một chiều lần lượt có dtype int8 và float64.
# Lưu ý: AUC/AUPRC yêu cầu dữ liệu có cả hai lớp và score phải hữu hạn.
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


# Mục đích: Tính ROC-AUC bằng thứ hạng, không phụ thuộc scikit-learn.
# Đầu vào: Label 0/1 và score dự đoán của từng mẫu.
# Đầu ra: Một số float trong khoảng [0, 1].
# Lưu ý: Các score bằng nhau nhận rank trung bình để xử lý tie đúng công thức.
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


# Mục đích: Tính Average Precision, tương đương diện tích bậc thang của PR curve.
# Đầu vào: Label 0/1 và score dự đoán của từng mẫu.
# Đầu ra: Giá trị AUPRC dạng float.
# Lưu ý: Mẫu được sắp theo score giảm dần và gộp các score bằng nhau.
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


# Mục đích: Tính toàn bộ metric báo cáo cho bài toán dropout nhị phân.
# Đầu vào: Label, xác suất lớp 1 và threshold phân lớp, mặc định 0.5.
# Đầu ra: Dictionary gồm AUC, AUPRC, F1, precision và recall.
# Lưu ý: Threshold chỉ ảnh hưởng F1/precision/recall, không ảnh hưởng AUC/AUPRC.
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
