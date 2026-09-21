# Tạo behavioral hyperedge bằng cosine kNN chỉ dùng train làm reference.
from __future__ import annotations

from pathlib import Path

import numpy as np


K_CANDIDATES = (5, 10, 20)
DEFAULT_K = 10
K_MAX = max(K_CANDIDATES)
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 100
HNSW_EF_SEARCH = 128
QUERY_BATCH_ROWS = 25_000


# Mục đích: Tạo bản sao float32 đã chuẩn hóa L2 theo từng node.
# Đầu vào: Matrix feature [nodes, features].
# Đầu ra: Matrix cùng shape; mỗi hàng khác 0 có norm bằng 1.
# Lưu ý: Hàng toàn 0 được giữ nguyên để tránh chia cho 0.
def _normalized_copy(matrix: np.ndarray) -> np.ndarray:
    output = np.asarray(matrix, dtype=np.float32).copy()
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    np.divide(output, norms, out=output, where=norms != 0)
    return output


# Mục đích: Kiểm tra neighbor matrix không vi phạm train-reference contract.
# Đầu vào: Matrix neighbor IDs, train mask và số neighbor kỳ vọng.
# Đầu ra: Không trả dữ liệu nếu mọi invariant hợp lệ.
# Lưu ý: Train anchor không được tự làm neighbor và một hàng không được trùng ID.
def validate_neighbors(
    neighbors: np.ndarray,
    train_mask: np.ndarray,
    *,
    k_max: int = K_MAX,
) -> None:
    if neighbors.ndim != 2 or neighbors.shape[1] != k_max:
        raise RuntimeError(f"Expected neighbor shape [nodes, {k_max}]")
    if np.any(neighbors < 0) or np.any(neighbors >= neighbors.shape[0]):
        raise RuntimeError("Behavioral neighbors contain an invalid node_id")
    if not np.all(train_mask[neighbors]):
        raise RuntimeError("Behavioral neighbors must all belong to the train split")
    if np.any(np.diff(np.sort(neighbors, axis=1), axis=1) == 0):
        raise RuntimeError("A behavioral edge contains duplicate neighbors")
    train_ids = np.flatnonzero(train_mask)
    if np.any(neighbors[train_ids] == train_ids[:, None]):
        raise RuntimeError("A train anchor cannot be its own neighbor")


# Mục đích: Tìm k cosine-nearest train nodes cho mọi node anchor.
# Đầu vào: Feature toàn bộ node, train_ids và k_max.
# Đầu ra: Matrix int32 [all_nodes, k_max] chứa global train node IDs.
# Lưu ý: FAISS index chỉ fit trên train; label không tham gia tìm neighbor.
def build_seed_neighbors(
    features: np.ndarray,
    train_ids: np.ndarray,
    *,
    k_max: int = K_MAX,
) -> np.ndarray:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("faiss-cpu is required to build behavioral hyperedges") from exc

    if features.ndim != 2 or features.shape[0] <= k_max:
        raise ValueError("features must be a 2-D matrix with more rows than k_max")
    train_ids = np.asarray(train_ids, dtype=np.int64)
    if train_ids.size <= k_max or np.unique(train_ids).size != train_ids.size:
        raise ValueError("train_ids must contain more than k_max unique nodes")

    normalized = _normalized_copy(features)
    train_features = np.ascontiguousarray(normalized[train_ids])
    index = faiss.IndexHNSWFlat(
        features.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT
    )
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(train_features)

    # Extra candidates allow removal of the anchor itself from train queries.
    search_k = min(k_max + 8, train_ids.size)
    output = np.empty((features.shape[0], k_max), dtype=np.int32)
    for start in range(0, features.shape[0], QUERY_BATCH_ROWS):
        stop = min(start + QUERY_BATCH_ROWS, features.shape[0])
        _, local_neighbors = index.search(
            np.ascontiguousarray(normalized[start:stop]), search_k
        )
        for offset, local_ids in enumerate(local_neighbors):
            anchor = start + offset
            global_ids = train_ids[local_ids]
            global_ids = global_ids[global_ids != anchor]
            if global_ids.size < k_max:
                raise RuntimeError(
                    f"Only {global_ids.size} valid neighbors found for node {anchor}"
                )
            output[anchor] = global_ids[:k_max]

    train_mask = np.zeros(features.shape[0], dtype=bool)
    train_mask[train_ids] = True
    validate_neighbors(output, train_mask, k_max=k_max)
    return output


# Mục đích: Ghi neighbor matrix thành NPZ nén theo cách atomic.
# Đầu vào: File đích và NumPy neighbor array.
# Đầu ra: Không trả dữ liệu; tạo file chứa key "neighbors".
def write_neighbors_atomic(path: Path, neighbors: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as destination:
        np.savez_compressed(destination, neighbors=neighbors)
    temporary.replace(path)
