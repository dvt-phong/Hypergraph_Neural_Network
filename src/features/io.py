# Đọc các block feature theo seed và node_id mà không nhân bản array lớn.
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import (
    DATASET_CONTRACT,
    DEFAULT_FEATURE_SET,
    EXPERIMENT_SEEDS,
    FEATURE_SETS,
    base_feature_columns,
    context_feature_columns,
    feature_columns,
    user_feature_columns,
)
from paths import PROCESSED_DATA_DIR, require_project_path


# Mục đích: Memory-map feature artifact và ghép đúng block theo feature_set.
# Đầu vào: Experiment seed, thư mục processed và tên feature_set.
# Đầu ra: Object cung cấp feature_names và hàm rows(node_ids).
# Lưu ý: Chỉ load X_context khi feature_set thực sự cần context.
class NodeFeatureStore:
    # Mục đích: Kiểm tra cấu hình và mở các file .npy ở chế độ memory map.
    # Đầu vào: Seed, output_dir và feature_set.
    # Đầu ra: Không trả riêng; lưu array và metadata vào object.
    # Lưu ý: Shape artifact được kiểm tra ngay để tránh lệch node/feature âm thầm.
    def __init__(
        self,
        seed: int,
        output_dir: Path = PROCESSED_DATA_DIR,
        *,
        feature_set: str = DEFAULT_FEATURE_SET,
    ) -> None:
        if seed not in EXPERIMENT_SEEDS:
            raise ValueError(f"seed must be one of {EXPERIMENT_SEEDS}")
        if feature_set not in FEATURE_SETS:
            raise ValueError(
                f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}"
            )
        self.seed = seed
        self.seed_index = EXPERIMENT_SEEDS.index(seed)
        self.feature_set = feature_set
        self.output_dir = require_project_path(output_dir)
        self.behavior = np.load(
            self.output_dir / "X_base.npy", mmap_mode="r"
        )
        expected_behavior = (
            len(EXPERIMENT_SEEDS),
            DATASET_CONTRACT.enrollments,
            len(base_feature_columns()),
        )
        if self.behavior.shape != expected_behavior:
            raise RuntimeError(f"Unexpected X_base shape: {self.behavior.shape}")
        self.context: np.ndarray | None = None
        if feature_set != "behavior":
            self.context = np.load(
                self.output_dir / "X_context.npy", mmap_mode="r"
            )
            expected_context = (
                len(EXPERIMENT_SEEDS),
                DATASET_CONTRACT.enrollments,
                len(context_feature_columns()),
            )
            if self.context.shape != expected_context:
                raise RuntimeError(
                    f"Unexpected X_context shape: {self.context.shape}"
                )

    @property
    # Mục đích: Trả schema cột đúng với feature_set hiện tại.
    # Đầu vào: Không có ngoài trạng thái object.
    # Đầu ra: Tuple tên feature theo đúng thứ tự matrix.
    def feature_names(self) -> tuple[str, ...]:
        return feature_columns(self.feature_set)

    # Mục đích: Lấy và ghép feature cho một danh sách global node_id.
    # Đầu vào: NumPy array node_ids theo thứ tự caller mong muốn.
    # Đầu ra: Matrix float32 [số node, số feature].
    # Lưu ý: Thứ tự output giữ nguyên thứ tự node_ids; không tự sort.
    def rows(self, node_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(node_ids, dtype=np.int64)
        behavior = np.asarray(self.behavior[self.seed_index, ids])
        if self.feature_set == "behavior":
            return behavior
        if self.context is None:
            raise RuntimeError("Context feature artifact was not loaded")
        context = self.context[self.seed_index, ids]
        user_stop = len(user_feature_columns())
        if self.feature_set == "behavior_user":
            selected = context[:, :user_stop]
        elif self.feature_set == "behavior_course":
            selected = context[:, user_stop:]
        else:
            selected = context
        features = np.concatenate((behavior, selected), axis=1)
        if features.shape[1] != len(self.feature_names):
            raise RuntimeError("Loaded node features do not match their schema")
        return features


# Mục đích: API ngắn để đọc feature mà không cần tự tạo NodeFeatureStore.
# Đầu vào: Seed, node_ids, output_dir và feature_set.
# Đầu ra: Matrix feature cho đúng các node được yêu cầu.
def load_node_features(
    seed: int,
    node_ids: np.ndarray,
    output_dir: Path = PROCESSED_DATA_DIR,
    *,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> np.ndarray:
    return NodeFeatureStore(
        seed, output_dir, feature_set=feature_set
    ).rows(node_ids)
