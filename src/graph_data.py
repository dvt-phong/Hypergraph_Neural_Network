# Đọc train graph và dựng local validation/test graph từ compact memberships.
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from scipy import sparse

from artifacts import sql_path
from config import DEFAULT_FEATURE_SET, EXPERIMENT_SEEDS
from features.io import NodeFeatureStore
from hypergraph.construction import (
    BEHAVIORAL_ARTIFACT,
    GRAPH_MANIFEST_ARTIFACT,
    HYPEREDGE_METADATA_ARTIFACT,
    TEST_MEMBERSHIPS_ARTIFACT,
    TRAIN_NODE_INDEX_ARTIFACT,
    VALIDATION_MEMBERSHIPS_ARTIFACT,
    train_matrix_name,
)
from paths import PROCESSED_DATA_DIR, require_project_path


@dataclass(frozen=True)
# Mục đích: Gom dữ liệu cần cho một full train-graph forward.
# Đầu vào: Global node IDs, feature matrix và CSR incidence H0.
# Đầu ra: Object bất biến được train loop sử dụng.
class TrainHypergraph:
    node_ids: np.ndarray
    features: np.ndarray
    incidence: sparse.csr_matrix


@dataclass(frozen=True)
# Mục đích: Biểu diễn graph inductive của đúng một validation/test target.
# Đầu vào: Node IDs/features, target mask, incidence và edge metadata.
# Đầu ra: Object độc lập; target chỉ kết nối tới train-reference nodes.
class LocalHypergraph:
    node_ids: np.ndarray
    features: np.ndarray
    target_mask: np.ndarray
    incidence: sparse.csr_matrix
    hyperedges: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
# Mục đích: Gom nhiều local graph thành một block-diagonal batch.
# Đầu vào: Các array đã concatenate và group IDs của node/edge.
# Đầu ra: Object dùng trực tiếp cho HGNN/HSL evaluation.
# Lưu ý: Group IDs giúp HSL không sample negative node từ graph khác.
class BatchedLocalHypergraph:
    node_ids: np.ndarray
    features: np.ndarray
    target_indices: np.ndarray
    incidence: sparse.csr_matrix
    node_graph_ids: np.ndarray
    edge_graph_ids: np.ndarray
    families: np.ndarray
    sizes: np.ndarray


# Mục đích: Chuyển experiment seed thành vị trí trong feature tensor.
# Đầu vào: Seed cần tìm.
# Đầu ra: Integer index trong EXPERIMENT_SEEDS.
# Lưu ý: Ném ValueError với seed ngoài protocol.
def _seed_index(seed: int) -> int:
    try:
        return EXPERIMENT_SEEDS.index(seed)
    except ValueError as exc:
        raise ValueError(f"seed must be one of {EXPERIMENT_SEEDS}") from exc


# Mục đích: Đọc global train node IDs theo đúng local row order của H0.
# Đầu vào: Kết nối, output directory và seed.
# Đầu ra: Vector int64 node IDs.
def _node_ids(
    connection: duckdb.DuckDBPyConnection,
    output_dir: Path,
    seed: int,
) -> np.ndarray:
    rows = connection.execute(
        f"""SELECT node_id
            FROM read_parquet('{sql_path(output_dir / TRAIN_NODE_INDEX_ARTIFACT)}')
            WHERE seed=? ORDER BY local_node_id""",
        [seed],
    ).fetchnumpy()
    return rows["node_id"].astype(np.int64, copy=False)


# Mục đích: Load train features và H0 với hàng được căn đúng node index.
# Đầu vào: Seed, output directory và feature_set.
# Đầu ra: TrainHypergraph gồm node_ids, features và CSR incidence.
# Lưu ý: Kiểm tra số hàng feature/incidence trước khi trả về.
def load_train_hypergraph(
    seed: int,
    output_dir: Path = PROCESSED_DATA_DIR,
    *,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> TrainHypergraph:
    _seed_index(seed)
    output_dir = require_project_path(output_dir)
    connection = duckdb.connect()
    try:
        node_ids = _node_ids(connection, output_dir, seed)
    finally:
        connection.close()
    incidence = sparse.load_npz(output_dir / train_matrix_name(seed)).tocsr()
    features = NodeFeatureStore(
        seed, output_dir, feature_set=feature_set
    ).rows(node_ids)
    if incidence.shape[0] != node_ids.size or features.shape[0] != node_ids.size:
        raise RuntimeError("Train feature and incidence rows do not match node index")
    return TrainHypergraph(node_ids, features, incidence)


# Mục đích: Ánh xạ tên experiment split sang artifact local membership.
# Đầu vào: "validation" hoặc "test".
# Đầu ra: Tên file Parquet tương ứng.
def _membership_name(experiment_split: str) -> str:
    if experiment_split == "validation":
        return VALIDATION_MEMBERSHIPS_ARTIFACT
    if experiment_split == "test":
        return TEST_MEMBERSHIPS_ARTIFACT
    raise ValueError("experiment_split must be validation or test")


# Mục đích: Giữ kết nối và artifact lớn mở để load nhiều local graph hiệu quả.
# Đầu vào: Seed, split, output directory và feature_set.
# Đầu ra: Context-manager object có target_node_ids() và load_many().
# Lưu ý: Phải close sau khi dùng; cú pháp with tự đóng connection.
class LocalGraphStore:
    columns = (
        "local_hyperedge_id",
        "family",
        "source_key",
        "object_type",
        "train_hyperedge_id",
        "singleton_reference_node_id",
        "behavioral_anchor_node_id",
        "size",
        "weight",
    )

    # Mục đích: Mở split membership, train H0, behavioral neighbors và features.
    # Đầu vào: Seed, validation/test split, output_dir và feature_set.
    # Đầu ra: Không trả riêng; lưu các reader/index vào object.
    # Lưu ý: Train incidence được chuyển CSC để lấy node của một edge nhanh.
    def __init__(
        self,
        seed: int,
        experiment_split: str,
        output_dir: Path = PROCESSED_DATA_DIR,
        *,
        feature_set: str = DEFAULT_FEATURE_SET,
    ) -> None:
        self.seed = seed
        self.seed_index = _seed_index(seed)
        self.experiment_split = experiment_split
        self.membership_name = _membership_name(experiment_split)
        self.output_dir = require_project_path(output_dir)
        self.feature_set = feature_set
        manifest = json.loads(
            (self.output_dir / GRAPH_MANIFEST_ARTIFACT).read_text(encoding="utf-8")
        )
        self.behavioral_k = int(manifest["behavioral_k"])
        self.connection = duckdb.connect()
        self.train_node_ids = _node_ids(self.connection, self.output_dir, seed)
        self.train_incidence = sparse.load_npz(
            self.output_dir / train_matrix_name(seed)
        ).tocsc()
        with np.load(
            self.output_dir / BEHAVIORAL_ARTIFACT, allow_pickle=False
        ) as archive:
            self.neighbors = archive["neighbors"][self.seed_index].copy()
        self.feature_store = NodeFeatureStore(
            seed, self.output_dir, feature_set=feature_set
        )

    # Mục đích: Đóng DuckDB connection của store.
    # Đầu vào: Không có.
    # Đầu ra: Không có.
    def close(self) -> None:
        self.connection.close()

    # Mục đích: Cho phép dùng LocalGraphStore trong câu lệnh with.
    # Đầu vào: Object hiện tại.
    # Đầu ra: Chính object LocalGraphStore.
    def __enter__(self) -> "LocalGraphStore":
        return self

    # Mục đích: Tự đóng connection khi rời context manager.
    # Đầu vào: Thông tin exception do Python truyền vào, nếu có.
    # Đầu ra: Không có; exception không bị nuốt.
    def __exit__(self, *_: object) -> None:
        self.close()

    # Mục đích: Liệt kê target node IDs có local membership trong split.
    # Đầu vào: Không có ngoài seed/split của store.
    # Đầu ra: Vector int64 target IDs đã sort.
    def target_node_ids(self) -> np.ndarray:
        rows = self.connection.execute(
            f"""SELECT DISTINCT target_node_id
                FROM read_parquet('{sql_path(self.output_dir / self.membership_name)}')
                WHERE seed=? ORDER BY target_node_id""",
            [self.seed],
        ).fetchnumpy()
        return rows["target_node_id"].astype(np.int64, copy=False)

    # Mục đích: Dựng nhiều local graph theo đúng thứ tự target được yêu cầu.
    # Đầu vào: Vector target_node_ids duy nhất và không rỗng.
    # Đầu ra: List LocalHypergraph cùng thứ tự với input.
    # Lưu ý: Query metadata một lần rồi assemble từng target để giảm I/O.
    def load_many(self, target_node_ids: np.ndarray) -> list[LocalHypergraph]:
        targets = np.asarray(target_node_ids, dtype=np.int64).reshape(-1)
        if targets.size == 0 or np.unique(targets).size != targets.size:
            raise ValueError("target_node_ids must be non-empty and unique")
        selected = self.connection.execute(
            f"""SELECT target_node_id, {', '.join(self.columns)}
                FROM read_parquet('{sql_path(self.output_dir / self.membership_name)}')
                WHERE seed=?
                  AND target_node_id IN (SELECT unnest(?::BIGINT[]))
                ORDER BY target_node_id, local_hyperedge_id""",
            [self.seed, targets.tolist()],
        ).fetchall()
        grouped: dict[int, list[tuple[Any, ...]]] = {
            int(target): [] for target in targets
        }
        for target, *values in selected:
            grouped[int(target)].append(tuple(values))
        missing = [target for target, rows in grouped.items() if not rows]
        if missing:
            raise ValueError(
                f"Targets are not in seed {self.seed} {self.experiment_split}: "
                f"{missing[:5]}"
            )
        return [self._assemble(int(target), grouped[int(target)]) for target in targets]

    # Mục đích: Chuyển các row compact membership thành một local sparse graph.
    # Đầu vào: Một target node ID và metadata rows của target đó.
    # Đầu ra: LocalHypergraph có target ở local row 0.
    # Lưu ý: Mọi node còn lại phải là train references; target không được lặp lại.
    def _assemble(
        self, target_node_id: int, rows: list[tuple[Any, ...]]
    ) -> LocalHypergraph:
        edge_references: list[np.ndarray] = []
        hyperedges: list[dict[str, Any]] = []
        for values in rows:
            edge = dict(zip(self.columns, values, strict=True))
            if edge["train_hyperedge_id"] is not None:
                local_rows = self.train_incidence.getcol(
                    int(edge["train_hyperedge_id"])
                ).indices
                references = self.train_node_ids[local_rows]
            elif edge["singleton_reference_node_id"] is not None:
                references = np.array(
                    [edge["singleton_reference_node_id"]], dtype=np.int64
                )
            elif edge["behavioral_anchor_node_id"] is not None:
                references = self.neighbors[
                    int(edge["behavioral_anchor_node_id"]), : self.behavioral_k
                ].astype(np.int64, copy=False)
            else:
                raise RuntimeError("Local hyperedge has no train-reference encoding")
            if references.size + 1 != edge["size"]:
                raise RuntimeError("Local hyperedge size does not match its references")
            edge_references.append(references)
            hyperedges.append(edge)

        all_references = np.unique(np.concatenate(edge_references))
        if np.any(all_references == target_node_id):
            raise RuntimeError("A target appears in its train reference set")
        node_ids = np.concatenate(
            (np.array([target_node_id], dtype=np.int64), all_references)
        )
        reference_to_local = {
            int(node_id): index + 1 for index, node_id in enumerate(all_references)
        }
        incidence_rows: list[int] = []
        incidence_columns: list[int] = []
        for column, references in enumerate(edge_references):
            incidence_rows.append(0)
            incidence_columns.append(column)
            incidence_rows.extend(
                reference_to_local[int(node)] for node in references
            )
            incidence_columns.extend([column] * references.size)
        incidence = sparse.coo_matrix(
            (
                np.ones(len(incidence_rows), dtype=np.uint8),
                (incidence_rows, incidence_columns),
            ),
            shape=(node_ids.size, len(edge_references)),
            dtype=np.uint8,
        ).tocsr()
        target_mask = np.zeros(node_ids.size, dtype=bool)
        target_mask[0] = True
        features = self.feature_store.rows(node_ids)
        return LocalHypergraph(
            node_ids=node_ids,
            features=features,
            target_mask=target_mask,
            incidence=incidence,
            hyperedges=tuple(hyperedges),
        )


# Mục đích: Ghép local graphs thành block-diagonal incidence batch.
# Đầu vào: List LocalHypergraph không rỗng.
# Đầu ra: BatchedLocalHypergraph chứa target offsets và node/edge group IDs.
# Lưu ý: Block diagonal bảo đảm các target không truyền message cho nhau.
def batch_local_hypergraphs(
    graphs: list[LocalHypergraph],
) -> BatchedLocalHypergraph:
    if not graphs:
        raise ValueError("At least one local graph is required")
    offsets = np.cumsum([0, *(graph.node_ids.size for graph in graphs[:-1])])
    incidence = sparse.block_diag(
        [graph.incidence for graph in graphs], format="csr", dtype=np.uint8
    )
    return BatchedLocalHypergraph(
        node_ids=np.concatenate([graph.node_ids for graph in graphs]),
        features=np.concatenate([graph.features for graph in graphs]),
        target_indices=offsets.astype(np.int64, copy=False),
        incidence=incidence,
        node_graph_ids=np.concatenate(
            [
                np.full(graph.node_ids.size, index, dtype=np.int64)
                for index, graph in enumerate(graphs)
            ]
        ),
        edge_graph_ids=np.concatenate(
            [
                np.full(graph.incidence.shape[1], index, dtype=np.int64)
                for index, graph in enumerate(graphs)
            ]
        ),
        families=np.concatenate(
            [
                np.asarray([edge["family"] for edge in graph.hyperedges])
                for graph in graphs
            ]
        ),
        sizes=np.asarray(incidence.sum(axis=0)).reshape(-1).astype(np.int64),
    )


# Mục đích: API tiện lợi để dựng đúng một local graph rồi đóng store ngay.
# Đầu vào: Seed, split, target ID, output directory và feature_set.
# Đầu ra: Một LocalHypergraph.
def load_local_hypergraph(
    seed: int,
    experiment_split: str,
    target_node_id: int,
    output_dir: Path = PROCESSED_DATA_DIR,
    *,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> LocalHypergraph:
    with LocalGraphStore(
        seed, experiment_split, output_dir, feature_set=feature_set
    ) as store:
        return store.load_many(np.array([target_node_id], dtype=np.int64))[0]
