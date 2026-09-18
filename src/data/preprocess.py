"""Materialize canonical XuetangX-247 Phase 1 artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from data.audit import (
    audit_dataset,
    csv_source,
    logs_sql,
    source_paths,
    truths_sql,
    validate_source_files,
)
from data.cache import signatures_match, source_signature, utc_now, write_json_atomic
from data.schema import DATASET_CONTRACT, SCHEMA_VERSION
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, require_project_path


PARQUET_ARTIFACTS = (
    "nodes.parquet",
    "events_35d.parquet",
    "users.parquet",
    "courses.parquet",
)


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _copy_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    destination: Path,
) -> None:
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    connection.execute(
        f"COPY ({query}) TO '{_sql_path(temporary)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)"
    )
    temporary.replace(destination)


def _cache_hit(raw_dir: Path, output_dir: Path) -> dict[str, Any] | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return None
    recorded_artifacts = {
        item.get("path"): item for item in manifest.get("artifacts", [])
    }
    for name in PARQUET_ARTIFACTS:
        path = output_dir / name
        item = recorded_artifacts.get(name)
        if not path.is_file() or item is None:
            return None
        if path.stat().st_size != item.get("size_bytes"):
            return None
    if not (output_dir / "audit.json").is_file():
        return None
    paths = source_paths(raw_dir)
    if not signatures_match(paths, manifest.get("source_files", [])):
        return None
    return manifest


def _artifact_record(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> dict[str, Any]:
    rows = connection.execute(
        f"SELECT count(*) FROM read_parquet('{_sql_path(path)}')"
    ).fetchone()[0]
    return {"path": path.name, "rows": rows, "size_bytes": path.stat().st_size}


def prepare_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    *,
    force: bool = False,
) -> dict[str, Any]:
    raw_dir = require_project_path(raw_dir)
    output_dir = require_project_path(output_dir)
    validate_source_files(raw_dir)
    paths = source_paths(raw_dir)
    initial_signatures = [source_signature(path, include_hash=False) for path in paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force and (cached := _cache_hit(raw_dir, output_dir)) is not None:
        return {**cached, "cache_hit": True}

    audit = audit_dataset(raw_dir, output_dir)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    logs = logs_sql(raw_dir)
    truths = truths_sql(raw_dir)
    raw_courses = csv_source(raw_dir / "course_info.csv")
    raw_users = csv_source(raw_dir / "user_info.csv")
    try:
        nodes_query = f"""
            WITH logs AS ({logs}), truth AS ({truths}),
            enrollment_meta AS (
                SELECT try_cast(enroll_id AS BIGINT) AS enroll_id,
                       try_cast(min(username) AS BIGINT) AS user_id,
                       min(course_id) AS course_id,
                       min(source_partition) AS source_partition
                FROM logs GROUP BY enroll_id
            ),
            labels AS (
                SELECT try_cast(enroll_id AS BIGINT) AS enroll_id,
                       min(source_partition) AS source_partition,
                       try_cast(min(truth) AS TINYINT) AS label
                FROM truth GROUP BY enroll_id
            )
            SELECT row_number() OVER (ORDER BY e.enroll_id)-1 AS node_id,
                   e.enroll_id, e.user_id, e.course_id,
                   e.source_partition, l.label
            FROM enrollment_meta e JOIN labels l USING(enroll_id, source_partition)
            ORDER BY e.enroll_id
        """
        _copy_parquet(connection, nodes_query, output_dir / "nodes.parquet")

        users_query = f"""
            WITH selected AS (
                SELECT DISTINCT user_id
                FROM read_parquet('{_sql_path(output_dir / 'nodes.parquet')}')
            ), users AS (
                SELECT try_cast(user_id AS BIGINT) AS user_id,
                       nullif(trim(gender), '') AS gender,
                       nullif(trim(education), '') AS education,
                       try_cast(birth AS DOUBLE)::INTEGER AS birth_year
                FROM {raw_users}
            )
            SELECT u.user_id, u.gender, u.education, u.birth_year
            FROM users u JOIN selected s USING(user_id)
            ORDER BY u.user_id
        """
        _copy_parquet(connection, users_query, output_dir / "users.parquet")

        courses_query = f"""
            WITH selected AS (
                SELECT DISTINCT course_id
                FROM read_parquet('{_sql_path(output_dir / 'nodes.parquet')}')
            ), courses AS (
                SELECT try_cast(id AS BIGINT) AS metadata_id,
                       course_id,
                       try_cast(start AS TIMESTAMP) AS course_start,
                       try_cast("end" AS TIMESTAMP) AS course_end,
                       try_cast(course_type AS TINYINT) AS course_type,
                       nullif(trim(category), '') AS category
                FROM {raw_courses}
            )
            SELECT c.* FROM courses c JOIN selected s USING(course_id)
            ORDER BY c.course_id
        """
        _copy_parquet(connection, courses_query, output_dir / "courses.parquet")

        events_query = f"""
            WITH logs AS ({logs}), parsed AS (
                SELECT try_cast(enroll_id AS BIGINT) AS enroll_id,
                       session_id, action,
                       nullif(trim(object), '') AS object_id,
                       try_cast(time AS TIMESTAMP) AS event_time
                FROM logs
            ), canonical AS (
                SELECT n.node_id, n.enroll_id, n.user_id, n.course_id,
                       n.source_partition, p.session_id, p.action, p.object_id,
                       p.event_time,
                       date_diff('day', cast(c.course_start AS DATE),
                                        cast(p.event_time AS DATE)) AS course_day
                FROM parsed p
                JOIN read_parquet('{_sql_path(output_dir / 'nodes.parquet')}') n
                  USING(enroll_id)
                JOIN read_parquet('{_sql_path(output_dir / 'courses.parquet')}') c
                  USING(course_id)
            )
            SELECT * FROM canonical WHERE course_day BETWEEN 0 AND 34
        """
        _copy_parquet(connection, events_query, output_dir / "events_35d.parquet")

        artifacts = [
            _artifact_record(connection, output_dir / name) for name in PARQUET_ARTIFACTS
        ]
    finally:
        connection.close()

    expected_rows = {
        "nodes.parquet": DATASET_CONTRACT.enrollments,
        "events_35d.parquet": DATASET_CONTRACT.retained_events_35d,
        "users.parquet": DATASET_CONTRACT.users,
        "courses.parquet": DATASET_CONTRACT.courses,
    }
    actual_rows = {item["path"]: item["rows"] for item in artifacts}
    if actual_rows != expected_rows:
        raise RuntimeError(
            f"Unexpected artifact rows: expected {expected_rows}, got {actual_rows}"
        )
    if not signatures_match(paths, initial_signatures):
        raise RuntimeError("A raw source file changed while Phase 1 artifacts were being built")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_CONTRACT.dataset,
        "generated_utc": utc_now(),
        "cache_hit": False,
        "source_files": [
            source_signature(path, include_hash=True) for path in paths
        ],
        "artifacts": artifacts,
        "audit_summary": {
            "raw_events": audit["logs"]["events"],
            "retained_events_35d": audit["temporal"]["retained_day_0_to_34"],
            "duplicate_rows_beyond_first": audit["duplicates"]["duplicate_rows_beyond_first"],
            "unknown_action_count": audit["logs"]["unknown_action_count"],
        },
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest
