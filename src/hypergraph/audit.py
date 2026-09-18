"""Audits for structural and behavioral hyperedge candidates."""
from __future__ import annotations

from typing import Any

import duckdb
import numpy as np

from data.cache import sql_path
from hypergraph.behavioral import K_CANDIDATES


def size_statistics(sizes: np.ndarray) -> dict[str, int | float]:
    """Summarize hyperedge cardinalities using deterministic percentiles."""

    sizes = np.asarray(sizes, dtype=np.int64)
    if sizes.size == 0:
        return {
            "hyperedges": 0,
            "median_size": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max_size": 0,
            "singletons": 0,
        }
    return {
        "hyperedges": int(sizes.size),
        "median_size": float(np.quantile(sizes, 0.50)),
        "p90": float(np.quantile(sizes, 0.90)),
        "p95": float(np.quantile(sizes, 0.95)),
        "p99": float(np.quantile(sizes, 0.99)),
        "max_size": int(sizes.max()),
        "singletons": int(np.count_nonzero(sizes == 1)),
    }


def _sizes(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    parameters: list[Any] | None = None,
) -> np.ndarray:
    rows = connection.execute(query, parameters or []).fetchnumpy()
    return rows["size"].astype(np.int64, copy=False)


def audit_structural(
    connection: duckdb.DuckDBPyConnection,
    memberships_path,
    splits_path,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    memberships = sql_path(memberships_path)
    splits = sql_path(splits_path)
    global_audit: dict[str, Any] = {}
    for family, keys in (
        ("course", "course_id"),
        ("object", "course_id, object_id, object_type"),
    ):
        values = _sizes(
            connection,
            f"""SELECT count(*)::BIGINT AS size
                FROM read_parquet('{memberships}') WHERE family='{family}'
                GROUP BY {keys}""",
        )
        global_audit[family] = size_statistics(values)
    global_audit["object_by_type"] = {}
    for object_type in ("video", "assignment", "forum"):
        values = _sizes(
            connection,
            f"""SELECT count(*)::BIGINT AS size
                FROM read_parquet('{memberships}')
                WHERE family='object' AND object_type=?
                GROUP BY course_id, object_id""",
            [object_type],
        )
        global_audit["object_by_type"][object_type] = size_statistics(values)

    train_by_seed: dict[str, Any] = {}
    for seed in seeds:
        families: dict[str, Any] = {}
        for family, keys in (
            ("course", "m.course_id"),
            ("object", "m.course_id, m.object_id, m.object_type"),
        ):
            values = _sizes(
                connection,
                f"""SELECT count(*)::BIGINT AS size
                    FROM read_parquet('{memberships}') m
                    JOIN read_parquet('{splits}') s USING (node_id)
                    WHERE m.family=? AND s.seed=? AND s.experiment_split='train'
                    GROUP BY {keys}""",
                [family, seed],
            )
            retained = values[values >= 2]
            families[family] = {
                **size_statistics(retained),
                "candidate_hyperedges": int(values.size),
                "removed_singletons": int(np.count_nonzero(values == 1)),
            }
        families["object_by_type"] = {}
        for object_type in ("video", "assignment", "forum"):
            values = _sizes(
                connection,
                f"""SELECT count(*)::BIGINT AS size
                    FROM read_parquet('{memberships}') m
                    JOIN read_parquet('{splits}') s USING (node_id)
                    WHERE m.family='object' AND m.object_type=?
                      AND s.seed=? AND s.experiment_split='train'
                    GROUP BY m.course_id, m.object_id""",
                [object_type, seed],
            )
            retained = values[values >= 2]
            families["object_by_type"][object_type] = {
                **size_statistics(retained),
                "candidate_hyperedges": int(values.size),
                "removed_singletons": int(np.count_nonzero(values == 1)),
            }
        train_by_seed[str(seed)] = families
    return {"global": global_audit, "train_by_seed": train_by_seed}


def audit_behavioral(
    neighbors: np.ndarray,
    train_ids_by_seed: list[np.ndarray],
) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    for seed_index, train_ids in enumerate(train_ids_by_seed):
        per_k: dict[str, Any] = {}
        for k in K_CANDIDATES:
            edges = np.concatenate(
                (train_ids[:, None], neighbors[seed_index, train_ids, :k]), axis=1
            )
            edges.sort(axis=1)
            unique_edges = np.unique(edges, axis=0)
            stats = size_statistics(
                np.full(unique_edges.shape[0], k + 1, dtype=np.int64)
            )
            per_k[str(k)] = {
                **stats,
                "anchors": int(train_ids.size),
                "duplicates_removed": int(train_ids.size - unique_edges.shape[0]),
            }
        by_seed[str(seed_index)] = per_k
    return by_seed
