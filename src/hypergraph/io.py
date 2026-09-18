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


def load_local_hypergraph(
    seed: int,
    experiment_split: str,
    target_node_id: int,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> LocalHypergraph:
    """Reconstruct one inductive graph containing one target and train references."""

    seed_index = _seed_index(seed)
    if experiment_split == "validation":
        membership_name = VALIDATION_MEMBERSHIPS_ARTIFACT
    elif experiment_split == "test":
        membership_name = TEST_MEMBERSHIPS_ARTIFACT
    else:
        raise ValueError("experiment_split must be validation or test")
    output_dir = require_project_path(output_dir)
    manifest = json.loads(
        (output_dir / GRAPH_MANIFEST_ARTIFACT).read_text(encoding="utf-8")
    )
    behavioral_k = int(manifest["behavioral_k"])

    connection = duckdb.connect()
    try:
        train_node_ids = _node_ids(connection, output_dir, seed)
        columns = [
            "local_hyperedge_id",
            "family",
            "source_key",
            "object_type",
            "train_hyperedge_id",
            "singleton_reference_node_id",
            "behavioral_anchor_node_id",
            "size",
            "weight",
        ]
        rows = connection.execute(
            f"""SELECT {', '.join(columns)}
                FROM read_parquet('{sql_path(output_dir / membership_name)}')
                WHERE seed=? AND target_node_id=?
                ORDER BY local_hyperedge_id""",
            [seed, target_node_id],
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(
            f"Target {target_node_id} is not in seed {seed} {experiment_split}"
        )

    train_incidence = sparse.load_npz(
        output_dir / train_matrix_name(seed)
    ).tocsc()
    with np.load(output_dir / BEHAVIORAL_ARTIFACT, allow_pickle=False) as archive:
        neighbors = archive["neighbors"]

    edge_references: list[np.ndarray] = []
    hyperedges: list[dict[str, Any]] = []
    for values in rows:
        edge = dict(zip(columns, values, strict=True))
        if edge["train_hyperedge_id"] is not None:
            local_rows = train_incidence.getcol(
                int(edge["train_hyperedge_id"])
            ).indices
            references = train_node_ids[local_rows]
        elif edge["singleton_reference_node_id"] is not None:
            references = np.array(
                [edge["singleton_reference_node_id"]], dtype=np.int64
            )
        elif edge["behavioral_anchor_node_id"] is not None:
            references = neighbors[
                seed_index, int(edge["behavioral_anchor_node_id"]), :behavioral_k
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
        incidence_rows.extend(reference_to_local[int(node)] for node in references)
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
    features = np.asarray(
        np.load(output_dir / "X_base.npy", mmap_mode="r")[seed_index, node_ids]
    )
    return LocalHypergraph(
        node_ids=node_ids,
        features=features,
        target_mask=target_mask,
        incidence=incidence,
        hyperedges=tuple(hyperedges),
    )
