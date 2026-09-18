"""Phase 4 orchestration for Course, Object and Behavioral hyperedges."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from data.cache import (
    copy_parquet_atomic,
    source_signature,
    sql_path,
    utc_now,
    write_json_atomic,
)
from data.schema import DATASET_CONTRACT, EXPERIMENT_SEEDS, SCHEMA_VERSION
from features.transform import build_features
from hypergraph.audit import audit_behavioral, audit_structural
from hypergraph.behavioral import (
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    K_CANDIDATES,
    K_MAX,
    build_seed_neighbors,
    write_neighbors_atomic,
)
from hypergraph.course import course_membership_query
from hypergraph.object import object_membership_query
from paths import PROCESSED_DATA_DIR, require_project_path


HYPEREDGE_VERSION = "hyperedge-families-v1"
STRUCTURAL_ARTIFACT = "structural_memberships.parquet"
BEHAVIORAL_ARTIFACT = "behavioral_neighbors.npz"
AUDIT_ARTIFACT = "hyperedge_audit.json"
MANIFEST_ARTIFACT = "hyperedge_manifest.json"


def _signatures(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    return {path.name: source_signature(path, include_hash=True) for path in paths}


def _cache_hit(
    output_dir: Path, inputs: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    path = output_dir / MANIFEST_ARTIFACT
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("hyperedge_version") != HYPEREDGE_VERSION
        or manifest.get("seeds") != list(EXPERIMENT_SEEDS)
        or manifest.get("k_candidates") != list(K_CANDIDATES)
        or manifest.get("inputs") != inputs
    ):
        return None
    artifacts = manifest.get("artifacts", {})
    for name in (STRUCTURAL_ARTIFACT, BEHAVIORAL_ARTIFACT, AUDIT_ARTIFACT):
        artifact = output_dir / name
        if not artifact.is_file() or artifacts.get(name) != source_signature(
            artifact, include_hash=True
        ):
            return None
    return manifest


def _build_structural(
    connection: duckdb.DuckDBPyConnection,
    nodes_path: Path,
    events_path: Path,
    destination: Path,
) -> None:
    query = f"""
        SELECT * FROM ({course_membership_query(nodes_path)})
        UNION ALL
        SELECT * FROM ({object_membership_query(events_path)})
        ORDER BY family, course_id, object_type, object_id, node_id
    """
    copy_parquet_atomic(connection, query, destination)


def _train_ids(
    connection: duckdb.DuckDBPyConnection, splits_path: Path, seed: int
) -> np.ndarray:
    result = connection.execute(
        f"""SELECT node_id FROM read_parquet('{sql_path(splits_path)}')
            WHERE seed=? AND experiment_split='train' ORDER BY node_id""",
        [seed],
    ).fetchnumpy()
    return result["node_id"].astype(np.int64, copy=False)


def build_hyperedges(
    output_dir: Path = PROCESSED_DATA_DIR,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build reusable Phase 4 candidates without reading labels."""

    output_dir = require_project_path(output_dir)
    build_features(output_dir=output_dir)
    nodes_path = output_dir / "nodes.parquet"
    events_path = output_dir / "events_35d.parquet"
    splits_path = output_dir / "splits.parquet"
    features_path = output_dir / "X_base.npy"
    input_paths = (nodes_path, events_path, splits_path, features_path)
    inputs = _signatures(input_paths)
    if not force and (cached := _cache_hit(output_dir, inputs)) is not None:
        return {**cached, "cache_hit": True}

    structural_path = output_dir / STRUCTURAL_ARTIFACT
    behavioral_path = output_dir / BEHAVIORAL_ARTIFACT
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    try:
        _build_structural(connection, nodes_path, events_path, structural_path)
        train_ids_by_seed = [
            _train_ids(connection, splits_path, seed) for seed in EXPERIMENT_SEEDS
        ]
        structural_audit = audit_structural(
            connection, structural_path, splits_path, EXPERIMENT_SEEDS
        )
    finally:
        connection.close()

    # One thread makes the approximate index construction reproducible.
    import faiss

    faiss.omp_set_num_threads(1)
    features = np.load(features_path, mmap_mode="r")
    neighbors = np.empty(
        (len(EXPERIMENT_SEEDS), DATASET_CONTRACT.enrollments, K_MAX),
        dtype=np.int32,
    )
    for seed_index, train_ids in enumerate(train_ids_by_seed):
        neighbors[seed_index] = build_seed_neighbors(
            features[seed_index], train_ids, k_max=K_MAX
        )
    write_neighbors_atomic(behavioral_path, neighbors)
    behavioral_audit_indexed = audit_behavioral(neighbors, train_ids_by_seed)
    behavioral_audit = {
        str(seed): behavioral_audit_indexed[str(index)]
        for index, seed in enumerate(EXPERIMENT_SEEDS)
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "hyperedge_version": HYPEREDGE_VERSION,
        "structural": structural_audit,
        "behavioral_train_by_seed": behavioral_audit,
        "rules": {
            "minimum_structural_cardinality": 2,
            "object_types": ["video", "assignment", "forum"],
            "behavioral_reference_split": "train only",
            "behavioral_metric": "cosine on X_base",
            "k_candidates": list(K_CANDIDATES),
        },
    }
    write_json_atomic(output_dir / AUDIT_ARTIFACT, audit)

    if _signatures(input_paths) != inputs:
        raise RuntimeError("An upstream artifact changed during Phase 4")
    artifact_paths = (
        structural_path,
        behavioral_path,
        output_dir / AUDIT_ARTIFACT,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "hyperedge_version": HYPEREDGE_VERSION,
        "dataset": DATASET_CONTRACT.dataset,
        "generated_utc": utc_now(),
        "cache_hit": False,
        "seeds": list(EXPERIMENT_SEEDS),
        "families": ["course", "object", "behavioral"],
        "k_candidates": list(K_CANDIDATES),
        "behavioral_array_layout": (
            "neighbors[seed_index, anchor_node_id, neighbor_rank]"
        ),
        "behavioral_array_shape": list(neighbors.shape),
        "behavioral_array_dtype": str(neighbors.dtype),
        "hnsw": {
            "metric": "cosine (L2-normalized inner product)",
            "M": HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION,
            "ef_search": HNSW_EF_SEARCH,
            "threads": 1,
        },
        "inputs": inputs,
        "artifacts": _signatures(artifact_paths),
    }
    write_json_atomic(output_dir / MANIFEST_ARTIFACT, manifest)
    return manifest
