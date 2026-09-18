"""Deterministic user-disjoint train/validation/test split."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import duckdb

from data.cache import (
    copy_parquet_atomic,
    source_signature,
    sql_path,
    utc_now,
    write_json_atomic,
)
from data.preprocess import prepare_dataset
from data.schema import DATASET_CONTRACT, EXPERIMENT_SPLITS, SCHEMA_VERSION
from paths import PROCESSED_DATA_DIR, require_project_path


SPLIT_VERSION = "user-disjoint-v1"
SPLIT_ARTIFACT = "splits.parquet"
SPLIT_MANIFEST = "split_manifest.json"
SPLIT_ALGORITHM = "stratified weighted round-robin group assignment"
MAX_RELATIVE_DEVIATION = 0.005


@dataclass(frozen=True)
class UserGroup:
    user_id: int
    enrollments: int
    dropouts: int


def integer_targets(
    total: int,
    ratios: tuple[float, ...],
    names: tuple[str, ...] = EXPERIMENT_SPLITS,
) -> dict[str, int]:
    """Allocate an integer total by largest remainder with stable tie-breaking."""

    if total < 0 or len(names) != len(ratios) or not names:
        raise ValueError("A non-negative total and matching non-empty ratios are required")
    if any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must be non-negative and sum to one")

    exact = [total * ratio for ratio in ratios]
    targets = [math.floor(value) for value in exact]
    remainder = total - sum(targets)
    order = sorted(
        range(len(names)),
        key=lambda index: (-(exact[index] - targets[index]), index),
    )
    for index in order[:remainder]:
        targets[index] += 1
    return dict(zip(names, targets, strict=True))


def _new_counts() -> dict[str, dict[str, int]]:
    return {
        split: {"users": 0, "enrollments": 0, "dropouts": 0}
        for split in EXPERIMENT_SPLITS
    }


def _target_counts(
    groups: list[UserGroup], ratios: tuple[float, float, float]
) -> dict[str, dict[str, int]]:
    totals = {
        "users": len(groups),
        "enrollments": sum(group.enrollments for group in groups),
        "dropouts": sum(group.dropouts for group in groups),
    }
    targets = _new_counts()
    for metric, total in totals.items():
        allocated = integer_targets(total, ratios)
        for split in EXPERIMENT_SPLITS:
            targets[split][metric] = allocated[split]
    return targets


def assign_user_groups(
    groups: list[UserGroup],
    *,
    ratios: tuple[float, float, float] = DATASET_CONTRACT.target_user_ratios,
    seed: int = DATASET_CONTRACT.split_seed,
) -> tuple[dict[int, str], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Assign whole user groups while balancing users, enrollments and labels."""

    if not groups:
        raise ValueError("At least one user group is required")
    if len({group.user_id for group in groups}) != len(groups):
        raise ValueError("Each user_id must appear exactly once")
    if any(
        group.enrollments <= 0
        or group.dropouts < 0
        or group.dropouts > group.enrollments
        for group in groups
    ):
        raise ValueError("Invalid enrollment or dropout count in a user group")

    targets = _target_counts(groups, ratios)
    actual = _new_counts()
    assignments: dict[int, str] = {}
    rng = random.Random(seed)
    ordered = list(groups)
    rng.shuffle(ordered)
    ordered.sort(
        key=lambda group: (
            group.enrollments,
            group.dropouts,
        ),
        reverse=True,
    )

    slot_counts = {split: 0 for split in EXPERIMENT_SPLITS}
    slots: list[str] = []
    for _ in ordered:
        candidates = [
            split
            for split in EXPERIMENT_SPLITS
            if slot_counts[split] < targets[split]["users"]
        ]
        split = min(
            candidates,
            key=lambda candidate: (
                (slot_counts[candidate] + 1) / targets[candidate]["users"],
                EXPERIMENT_SPLITS.index(candidate),
            ),
        )
        slots.append(split)
        slot_counts[split] += 1

    for group, split in zip(ordered, slots, strict=True):
        assignments[group.user_id] = split
        actual[split]["users"] += 1
        actual[split]["enrollments"] += group.enrollments
        actual[split]["dropouts"] += group.dropouts

    if any(actual[split]["users"] != targets[split]["users"] for split in EXPERIMENT_SPLITS):
        raise RuntimeError("The assignment did not meet exact user targets")
    return assignments, targets, actual


def _load_user_groups(
    connection: duckdb.DuckDBPyConnection, nodes_path: Path
) -> list[UserGroup]:
    rows = connection.execute(
        f"""
        SELECT user_id, count(*) AS enrollments,
               sum(CASE WHEN label=1 THEN 1 ELSE 0 END) AS dropouts
        FROM read_parquet('{sql_path(nodes_path)}')
        GROUP BY user_id ORDER BY user_id
        """
    ).fetchall()
    return [UserGroup(int(row[0]), int(row[1]), int(row[2])) for row in rows]


def _split_cache_hit(
    output_dir: Path,
    nodes_signature: dict[str, Any],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, Any] | None:
    artifact_path = output_dir / SPLIT_ARTIFACT
    manifest_path = output_dir / SPLIT_MANIFEST
    if not artifact_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("split_version") != SPLIT_VERSION
        or manifest.get("algorithm") != SPLIT_ALGORITHM
        or manifest.get("seed") != seed
        or manifest.get("max_relative_deviation") != MAX_RELATIVE_DEVIATION
        or manifest.get("target_ratios")
        != dict(zip(EXPERIMENT_SPLITS, ratios, strict=True))
        or manifest.get("input_nodes") != nodes_signature
    ):
        return None
    artifact = manifest.get("artifact", {})
    if artifact_path.stat().st_size != artifact.get("size_bytes"):
        return None
    if source_signature(artifact_path, include_hash=True) != artifact:
        return None
    return manifest


def _validate_and_summarize(
    connection: duckdb.DuckDBPyConnection,
    nodes_path: Path,
    split_path: Path,
    targets: dict[str, dict[str, int]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    split_source = f"read_parquet('{sql_path(split_path)}')"
    nodes_source = f"read_parquet('{sql_path(nodes_path)}')"
    invariant_row = connection.execute(
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT node_id) AS node_ids,
               count(DISTINCT enroll_id) AS enrollments,
               count(DISTINCT user_id) AS users,
               count(*) FILTER (
                   WHERE node_id IS NULL OR enroll_id IS NULL OR user_id IS NULL
                      OR source_partition IS NULL OR experiment_split IS NULL
               ) AS missing_required,
               count(*) FILTER (
                   WHERE experiment_split NOT IN ('train', 'validation', 'test')
               ) AS unknown_split
        FROM {split_source}
        """
    ).fetchone()
    overlap = connection.execute(
        f"""
        SELECT count(*) FROM (
            SELECT user_id FROM {split_source}
            GROUP BY user_id HAVING count(DISTINCT experiment_split)>1
        )
        """
    ).fetchone()[0]
    invariants = {
        "rows": int(invariant_row[0]),
        "distinct_node_ids": int(invariant_row[1]),
        "distinct_enrollments": int(invariant_row[2]),
        "distinct_users": int(invariant_row[3]),
        "missing_required": int(invariant_row[4]),
        "unknown_split": int(invariant_row[5]),
        "user_overlap": int(overlap),
    }
    expected = DATASET_CONTRACT
    if invariants != {
        "rows": expected.enrollments,
        "distinct_node_ids": expected.enrollments,
        "distinct_enrollments": expected.enrollments,
        "distinct_users": expected.users,
        "missing_required": 0,
        "unknown_split": 0,
        "user_overlap": 0,
    }:
        raise RuntimeError(f"Split invariant failed: {invariants}")

    rows = connection.execute(
        f"""
        SELECT s.experiment_split,
               count(DISTINCT s.user_id) AS users,
               count(*) AS enrollments,
               sum(CASE WHEN n.label=1 THEN 1 ELSE 0 END) AS dropouts,
               sum(CASE WHEN n.label=0 THEN 1 ELSE 0 END) AS non_dropouts,
               count(*) FILTER (WHERE s.source_partition='train') AS source_train,
               count(*) FILTER (WHERE s.source_partition='test') AS source_test
        FROM {split_source} s JOIN {nodes_source} n USING(node_id)
        GROUP BY s.experiment_split
        """
    ).fetchall()
    row_by_split = {row[0]: row for row in rows}
    if set(row_by_split) != set(EXPERIMENT_SPLITS):
        missing = set(EXPERIMENT_SPLITS) - set(row_by_split)
        raise RuntimeError(f"Missing experiment split: {missing}")

    summary: dict[str, dict[str, Any]] = {}
    for split in EXPERIMENT_SPLITS:
        row = row_by_split[split]
        users = int(row[1])
        enrollments = int(row[2])
        dropouts = int(row[3])
        target = targets[split]
        if users != target["users"]:
            raise RuntimeError(f"Unexpected user count in {split}: {users}")
        enrollment_deviation = enrollments - target["enrollments"]
        dropout_deviation = dropouts - target["dropouts"]
        if (
            abs(enrollment_deviation) / target["enrollments"]
            >= MAX_RELATIVE_DEVIATION
            or abs(dropout_deviation) / target["dropouts"]
            >= MAX_RELATIVE_DEVIATION
        ):
            raise RuntimeError(
                f"Split balance tolerance exceeded in {split}: "
                f"enrollments={enrollment_deviation}, dropouts={dropout_deviation}"
            )
        summary[split] = {
            "users": users,
            "target_users": target["users"],
            "enrollments": enrollments,
            "target_enrollments": target["enrollments"],
            "enrollment_deviation": enrollment_deviation,
            "dropouts": dropouts,
            "target_dropouts": target["dropouts"],
            "dropout_deviation": dropout_deviation,
            "non_dropouts": int(row[4]),
            "dropout_rate": dropouts / enrollments,
            "source_train_enrollments": int(row[5]),
            "source_test_enrollments": int(row[6]),
        }
    return summary, invariants


def build_splits(
    output_dir: Path = PROCESSED_DATA_DIR,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build and cache the locked user-disjoint experiment split."""

    output_dir = require_project_path(output_dir)
    prepare_dataset(output_dir=output_dir)
    nodes_path = output_dir / "nodes.parquet"
    split_path = output_dir / SPLIT_ARTIFACT
    ratios = DATASET_CONTRACT.target_user_ratios
    seed = DATASET_CONTRACT.split_seed
    nodes_signature = source_signature(nodes_path, include_hash=True)
    if not force and (
        cached := _split_cache_hit(output_dir, nodes_signature, ratios, seed)
    ) is not None:
        return {**cached, "cache_hit": True}

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    try:
        groups = _load_user_groups(connection, nodes_path)
        assignments, targets, actual = assign_user_groups(
            groups, ratios=ratios, seed=seed
        )
        assignment_rows = sorted(assignments.items())
        connection.execute(
            "CREATE TEMP TABLE user_splits "
            "(user_id BIGINT PRIMARY KEY, experiment_split VARCHAR NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO user_splits VALUES (?, ?)", assignment_rows
        )
        query = f"""
            SELECT n.node_id, n.enroll_id, n.user_id, n.source_partition,
                   s.experiment_split
            FROM read_parquet('{sql_path(nodes_path)}') n
            JOIN user_splits s USING(user_id)
            ORDER BY n.node_id
        """
        copy_parquet_atomic(connection, query, split_path)
        summary, invariants = _validate_and_summarize(
            connection, nodes_path, split_path, targets
        )
    finally:
        connection.close()

    if source_signature(nodes_path, include_hash=True) != nodes_signature:
        raise RuntimeError("nodes.parquet changed while the split was being built")
    objective = sum(
        ((actual[split][metric] - targets[split][metric]) / targets[split][metric])
        ** 2
        for split in EXPERIMENT_SPLITS
        for metric in ("users", "enrollments", "dropouts")
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split_version": SPLIT_VERSION,
        "dataset": DATASET_CONTRACT.dataset,
        "generated_utc": utc_now(),
        "cache_hit": False,
        "seed": seed,
        "algorithm": SPLIT_ALGORITHM,
        "balanced_metrics": ["users", "enrollments", "dropouts"],
        "max_relative_deviation": MAX_RELATIVE_DEVIATION,
        "target_ratios": dict(zip(EXPERIMENT_SPLITS, ratios, strict=True)),
        "objective": objective,
        "input_nodes": nodes_signature,
        "artifact": source_signature(split_path, include_hash=True),
        "invariants": invariants,
        "summary": summary,
    }
    write_json_atomic(output_dir / SPLIT_MANIFEST, manifest)
    return manifest
