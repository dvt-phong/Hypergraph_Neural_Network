"""Leakage-safe transformation and caching of the 60-dimensional X_base."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np

from data.cache import source_signature, sql_path, utc_now, write_json_atomic
from data.schema import (
    DATASET_CONTRACT,
    EXPERIMENT_SEEDS,
    SCHEMA_VERSION,
    base_feature_columns,
)
from data.split import build_splits
from features.engineering import ABLATION_FEATURES, build_raw_features
from paths import PROCESSED_DATA_DIR, require_project_path


FEATURE_VERSION = "x-base-60-v1"
RAW_FEATURES_ARTIFACT = "features_raw.parquet"
X_BASE_ARTIFACT = "X_base.npy"
TRANSFORM_ARTIFACT = "feature_transform.joblib"
FEATURE_MANIFEST = "feature_manifest.json"
TRANSFORM_CHUNK_ROWS = 50_000


def _signatures(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    return {path.name: source_signature(path, include_hash=True) for path in paths}


def _feature_cache_hit(
    output_dir: Path,
    input_signatures: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    manifest_path = output_dir / FEATURE_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("feature_version") != FEATURE_VERSION
        or manifest.get("seeds") != list(EXPERIMENT_SEEDS)
        or manifest.get("feature_names") != list(base_feature_columns())
        or manifest.get("inputs") != input_signatures
    ):
        return None
    recorded = manifest.get("artifacts", {})
    for name in (RAW_FEATURES_ARTIFACT, X_BASE_ARTIFACT, TRANSFORM_ARTIFACT):
        path = output_dir / name
        if not path.is_file() or recorded.get(name) is None:
            return None
        if source_signature(path, include_hash=True) != recorded[name]:
            return None
    return manifest


def _load_raw_matrix(
    connection: duckdb.DuckDBPyConnection,
    features_path: Path,
) -> np.ndarray:
    columns = base_feature_columns()
    selected = ", ".join(columns)
    arrays = connection.execute(
        f"SELECT {selected} FROM read_parquet('{sql_path(features_path)}') "
        "ORDER BY node_id"
    ).fetchnumpy()
    matrix = np.empty(
        (DATASET_CONTRACT.enrollments, len(columns)), dtype=np.float32
    )
    for index, column in enumerate(columns):
        matrix[:, index] = arrays[column]
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise RuntimeError("Raw X_base contains non-finite or negative values")
    np.log1p(matrix, out=matrix)
    return matrix


def _train_node_ids(
    connection: duckdb.DuckDBPyConnection,
    splits_path: Path,
    seed: int,
) -> np.ndarray:
    result = connection.execute(
        f"""
        SELECT node_id FROM read_parquet('{sql_path(splits_path)}')
        WHERE seed=? AND experiment_split='train' ORDER BY node_id
        """,
        [seed],
    ).fetchnumpy()
    return result["node_id"].astype(np.int64, copy=False)


def _write_x_base(
    connection: duckdb.DuckDBPyConnection,
    logged_matrix: np.ndarray,
    splits_path: Path,
    destination: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    shape = (
        len(EXPERIMENT_SEEDS),
        DATASET_CONTRACT.enrollments,
        len(base_feature_columns()),
    )
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=shape
    )
    transform_by_seed: dict[str, Any] = {}
    audit_by_seed: dict[str, Any] = {}
    try:
        for seed_index, seed in enumerate(EXPERIMENT_SEEDS):
            train_ids = _train_node_ids(connection, splits_path, seed)
            train_matrix = logged_matrix[train_ids]
            mean = np.mean(train_matrix, axis=0, dtype=np.float64)
            standard_deviation = np.std(train_matrix, axis=0, dtype=np.float64)
            zero_variance = standard_deviation == 0
            scale = standard_deviation.copy()
            scale[zero_variance] = 1.0
            mean32 = mean.astype(np.float32)
            scale32 = scale.astype(np.float32)

            for start in range(0, logged_matrix.shape[0], TRANSFORM_CHUNK_ROWS):
                stop = min(start + TRANSFORM_CHUNK_ROWS, logged_matrix.shape[0])
                output[seed_index, start:stop] = (
                    logged_matrix[start:stop] - mean32
                ) / scale32

            transformed_train = (train_matrix - mean32) / scale32
            variable = ~zero_variance
            train_mean = np.mean(transformed_train, axis=0, dtype=np.float64)
            train_std = np.std(transformed_train, axis=0, dtype=np.float64)
            max_mean_error = (
                float(np.max(np.abs(train_mean[variable])))
                if np.any(variable)
                else 0.0
            )
            max_std_error = (
                float(np.max(np.abs(train_std[variable] - 1.0)))
                if np.any(variable)
                else 0.0
            )
            zero_variance_features = [
                name
                for name, is_zero in zip(
                    base_feature_columns(), zero_variance, strict=True
                )
                if is_zero
            ]
            transform_by_seed[str(seed)] = {
                "mean": mean32,
                "scale": scale32,
                "train_nodes": int(train_ids.size),
                "zero_variance_features": zero_variance_features,
            }
            audit_by_seed[str(seed)] = {
                "train_nodes": int(train_ids.size),
                "max_abs_train_mean": max_mean_error,
                "max_abs_train_std_error": max_std_error,
                "zero_variance_features": zero_variance_features,
            }
            del train_matrix, transformed_train
        output.flush()
    finally:
        del output
    temporary.replace(destination)
    transform = {
        "feature_version": FEATURE_VERSION,
        "feature_names": list(base_feature_columns()),
        "operation": "log1p followed by train-only standardization",
        "seed_order": list(EXPERIMENT_SEEDS),
        "by_seed": transform_by_seed,
    }
    return transform, audit_by_seed


def _write_transform(path: Path, transform: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    joblib.dump(transform, temporary, compress=3)
    temporary.replace(path)


def build_features(
    output_dir: Path = PROCESSED_DATA_DIR,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build raw features and train-only normalized X_base for all five seeds."""

    output_dir = require_project_path(output_dir)
    build_splits(output_dir=output_dir)
    nodes_path = output_dir / "nodes.parquet"
    events_path = output_dir / "events_35d.parquet"
    splits_path = output_dir / "splits.parquet"
    input_paths = (nodes_path, events_path, splits_path)
    input_signatures = _signatures(input_paths)
    if not force and (
        cached := _feature_cache_hit(output_dir, input_signatures)
    ) is not None:
        return {**cached, "cache_hit": True}

    raw_features_path = output_dir / RAW_FEATURES_ARTIFACT
    x_base_path = output_dir / X_BASE_ARTIFACT
    transform_path = output_dir / TRANSFORM_ARTIFACT
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    try:
        raw_audit = build_raw_features(
            connection, nodes_path, events_path, raw_features_path
        )
        logged_matrix = _load_raw_matrix(connection, raw_features_path)
        transform, transform_audit = _write_x_base(
            connection, logged_matrix, splits_path, x_base_path
        )
        del logged_matrix
    finally:
        connection.close()
    _write_transform(transform_path, transform)

    if _signatures(input_paths) != input_signatures:
        raise RuntimeError("A Phase 1/2 artifact changed while features were being built")
    artifacts = _signatures((raw_features_path, x_base_path, transform_path))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "dataset": DATASET_CONTRACT.dataset,
        "generated_utc": utc_now(),
        "cache_hit": False,
        "seeds": list(EXPERIMENT_SEEDS),
        "feature_names": list(base_feature_columns()),
        "ablation_features": list(ABLATION_FEATURES),
        "transform": "log1p followed by train-only standardization",
        "array_layout": "X_base[seed_index, node_id, feature_index]",
        "array_shape": [
            len(EXPERIMENT_SEEDS),
            DATASET_CONTRACT.enrollments,
            len(base_feature_columns()),
        ],
        "array_dtype": "float32",
        "inputs": input_signatures,
        "artifacts": artifacts,
        "raw_feature_audit": raw_audit,
        "transform_audit_by_seed": transform_audit,
    }
    write_json_atomic(output_dir / FEATURE_MANIFEST, manifest)
    return manifest
