"""Load materialized train graphs and reconstruct compact local graphs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from scipy import sparse

from data.cache import sql_path
from data.schema import EXPERIMENT_SEEDS
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
class TrainHypergraph:
    node_ids: np.ndarray
    features: np.ndarray
    incidence: sparse.csr_matrix


@dataclass(frozen=True)
class LocalHypergraph:
    node_ids: np.ndarray
    features: np.ndarray
    target_mask: np.ndarray
    incidence: sparse.csr_matrix
    hyperedges: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BatchedLocalHypergraph:
    node_ids: np.ndarray
    features: np.ndarray
    target_indices: np.ndarray
    incidence: sparse.csr_matrix


def _seed_index(seed: int) -> int:
    try:
        return EXPERIMENT_SEEDS.index(seed)
    except ValueError as exc:
        raise ValueError(f"seed must be one of {EXPERIMENT_SEEDS}") from exc


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


def load_train_hypergraph(
    seed: int,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> TrainHypergraph:
    """Load train features and H0 with the same local node-row ordering."""

    seed_index = _seed_index(seed)
    output_dir = require_project_path(output_dir)
    connection = duckdb.connect()
    try:
        node_ids = _node_ids(connection, output_dir, seed)
    finally:
        connection.close()
    incidence = sparse.load_npz(output_dir / train_matrix_name(seed)).tocsr()
    features = np.asarray(
        np.load(output_dir / "X_base.npy", mmap_mode="r")[seed_index, node_ids]
    )
    if incidence.shape[0] != node_ids.size or features.shape[0] != node_ids.size:
        raise RuntimeError("Train feature and incidence rows do not match node index")
    return TrainHypergraph(node_ids, features, incidence)


def _membership_name(experiment_split: str) -> str:
    if experiment_split == "validation":
        return VALIDATION_MEMBERSHIPS_ARTIFACT
    if experiment_split == "test":
        return TEST_MEMBERSHIPS_ARTIFACT
    raise ValueError("experiment_split must be validation or test")


class LocalGraphStore:
    """Reuse large Phase 5 artifacts while loading many independent targets."""

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

    def __init__(
        self,
        seed: int,
        experiment_split: str,
        output_dir: Path = PROCESSED_DATA_DIR,
    ) -> None:
        self.seed = seed
        self.seed_index = _seed_index(seed)
        self.experiment_split = experiment_split
        self.membership_name = _membership_name(experiment_split)
        self.output_dir = require_project_path(output_dir)
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
        self.features = np.load(self.output_dir / "X_base.npy", mmap_mode="r")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "LocalGraphStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def target_node_ids(self) -> np.ndarray:
        rows = self.connection.execute(
            f"""SELECT DISTINCT target_node_id
                FROM read_parquet('{sql_path(self.output_dir / self.membership_name)}')
                WHERE seed=? ORDER BY target_node_id""",
            [self.seed],
        ).fetchnumpy()
        return rows["target_node_id"].astype(np.int64, copy=False)

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
        features = np.asarray(self.features[self.seed_index, node_ids])
        return LocalHypergraph(
            node_ids=node_ids,
            features=features,
            target_mask=target_mask,
            incidence=incidence,
            hyperedges=tuple(hyperedges),
        )


def batch_local_hypergraphs(
    graphs: list[LocalHypergraph],
) -> BatchedLocalHypergraph:
    """Create a block-diagonal batch; graphs cannot exchange messages."""

    if not graphs:
        raise ValueError("At least one local graph is required")
    offsets = np.cumsum([0, *(graph.node_ids.size for graph in graphs[:-1])])
    return BatchedLocalHypergraph(
        node_ids=np.concatenate([graph.node_ids for graph in graphs]),
        features=np.concatenate([graph.features for graph in graphs]),
        target_indices=offsets.astype(np.int64, copy=False),
        incidence=sparse.block_diag(
            [graph.incidence for graph in graphs], format="csr", dtype=np.uint8
        ),
    )


def load_local_hypergraph(
    seed: int,
    experiment_split: str,
    target_node_id: int,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> LocalHypergraph:
    """Reconstruct one inductive graph containing one target and train references."""

    with LocalGraphStore(seed, experiment_split, output_dir) as store:
        return store.load_many(np.array([target_node_id], dtype=np.int64))[0]
