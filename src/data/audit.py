"""Source validation and full audit for the labeled XuetangX-247 data."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import duckdb

from data.cache import sql_path, utc_now, write_json_atomic
from data.schema import (
    ACTIONS,
    COURSE_COLUMNS,
    DATASET_CONTRACT,
    LOG_COLUMNS,
    SCHEMA_VERSION,
    TRUTH_COLUMNS,
    USER_COLUMNS,
)
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, require_project_path


SOURCE_SCHEMAS = {
    "train_log.csv": LOG_COLUMNS,
    "test_log.csv": LOG_COLUMNS,
    "train_truth.csv": TRUTH_COLUMNS,
    "test_truth.csv": TRUTH_COLUMNS,
    "user_info.csv": USER_COLUMNS,
    "course_info.csv": COURSE_COLUMNS,
}


class DataAuditError(RuntimeError):
    """Raised when the raw dataset violates the locked contract."""


def source_paths(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    return [raw_dir / name for name in SOURCE_SCHEMAS]


def _read_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return tuple(next(csv.reader(source)))


def validate_source_files(raw_dir: Path = RAW_DATA_DIR) -> dict[str, dict[str, Any]]:
    raw_dir = require_project_path(raw_dir)
    results: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for filename, expected_header in SOURCE_SCHEMAS.items():
        path = raw_dir / filename
        if not path.is_file():
            errors.append(f"Missing source file: {path}")
            continue
        actual_header = _read_header(path)
        if actual_header != expected_header:
            errors.append(
                f"Unexpected header in {filename}: expected {expected_header}, "
                f"got {actual_header}"
            )
        results[filename] = {
            "size_bytes": path.stat().st_size,
            "columns": list(actual_header),
        }
    if errors:
        raise DataAuditError("\n".join(errors))
    return results


def csv_source(path: Path) -> str:
    return (
        f"read_csv_auto('{sql_path(path)}', header=true, "
        "all_varchar=true, nullstr='')"
    )


def logs_sql(raw_dir: Path) -> str:
    train = csv_source(raw_dir / "train_log.csv")
    test = csv_source(raw_dir / "test_log.csv")
    return f"""
        SELECT *, 'train' AS source_partition FROM {train}
        UNION ALL
        SELECT *, 'test' AS source_partition FROM {test}
    """


def truths_sql(raw_dir: Path) -> str:
    train = csv_source(raw_dir / "train_truth.csv")
    test = csv_source(raw_dir / "test_truth.csv")
    return f"""
        SELECT *, 'train' AS source_partition FROM {train}
        UNION ALL
        SELECT *, 'test' AS source_partition FROM {test}
    """


def _tuple(columns: list[str], row: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(columns, row, strict=True))


def run_audit(connection: duckdb.DuckDBPyConnection, raw_dir: Path) -> dict[str, Any]:
    raw_dir = require_project_path(raw_dir)
    action_values = ", ".join("'" + action.replace("'", "''") + "'" for action in ACTIONS)
    logs = logs_sql(raw_dir)
    truths = truths_sql(raw_dir)
    courses = csv_source(raw_dir / "course_info.csv")
    users = csv_source(raw_dir / "user_info.csv")

    log_columns = [
        "events",
        "enrollments",
        "users",
        "courses",
        "actions",
        "missing_enroll_id",
        "missing_user",
        "missing_course",
        "missing_session",
        "missing_action",
        "missing_object",
        "missing_time",
        "invalid_enroll_id",
        "invalid_user_id",
        "unknown_action_count",
        "invalid_timestamp",
    ]
    log_row = connection.execute(
        f"""
        WITH logs AS ({logs})
        SELECT
            count(*),
            count(DISTINCT enroll_id),
            count(DISTINCT username),
            count(DISTINCT course_id),
            count(DISTINCT action),
            count(*) FILTER (WHERE enroll_id IS NULL OR trim(enroll_id)=''),
            count(*) FILTER (WHERE username IS NULL OR trim(username)=''),
            count(*) FILTER (WHERE course_id IS NULL OR trim(course_id)=''),
            count(*) FILTER (WHERE session_id IS NULL OR trim(session_id)=''),
            count(*) FILTER (WHERE action IS NULL OR trim(action)=''),
            count(*) FILTER (WHERE object IS NULL OR trim(object)=''),
            count(*) FILTER (WHERE time IS NULL OR trim(time)=''),
            count(*) FILTER (
                WHERE enroll_id IS NOT NULL AND try_cast(enroll_id AS BIGINT) IS NULL
            ),
            count(*) FILTER (
                WHERE username IS NOT NULL AND try_cast(username AS BIGINT) IS NULL
            ),
            count(*) FILTER (WHERE action NOT IN ({action_values})),
            count(*) FILTER (WHERE try_cast(time AS TIMESTAMP) IS NULL)
        FROM logs
        """
    ).fetchone()

    split_rows = connection.execute(
        f"""
        WITH logs AS ({logs})
        SELECT source_partition, count(*) AS events,
               count(DISTINCT enroll_id) AS enrollments,
               count(DISTINCT username) AS users,
               count(DISTINCT course_id) AS courses
        FROM logs GROUP BY source_partition ORDER BY source_partition
        """
    ).fetchall()

    truth_rows = connection.execute(
        f"""
        WITH truth AS ({truths})
        SELECT source_partition, count(*) AS labels,
               count(DISTINCT enroll_id) AS enrollments,
               count(*) FILTER (WHERE try_cast(truth AS INTEGER)=1) AS dropout,
               count(*) FILTER (WHERE try_cast(truth AS INTEGER)=0) AS non_dropout,
               count(*) FILTER (
                   WHERE try_cast(truth AS INTEGER) IS NULL
                      OR try_cast(truth AS INTEGER) NOT IN (0,1)
               ) AS invalid
        FROM truth GROUP BY source_partition ORDER BY source_partition
        """
    ).fetchall()

    relation_columns = [
        "log_enrollments_without_truth",
        "truth_enrollments_without_log",
        "source_partition_mismatches",
        "conflicting_enrollment_metadata",
        "duplicate_user_course_pairs",
    ]
    relation_row = connection.execute(
        f"""
        WITH logs AS ({logs}), truth AS ({truths}),
        enrollment_meta AS (
            SELECT enroll_id,
                   count(DISTINCT username) AS users,
                   count(DISTINCT course_id) AS courses,
                   count(DISTINCT source_partition) AS partitions,
                   min(username) AS username,
                   min(course_id) AS course_id,
                   min(source_partition) AS source_partition
            FROM logs GROUP BY enroll_id
        ),
        truth_meta AS (
            SELECT enroll_id, count(DISTINCT source_partition) AS partitions,
                   min(source_partition) AS source_partition
            FROM truth GROUP BY enroll_id
        ),
        pairs AS (
            SELECT username, course_id, count(*) AS enrollments
            FROM enrollment_meta GROUP BY username, course_id
        )
        SELECT
            (SELECT count(*) FROM enrollment_meta e
             LEFT JOIN truth_meta t USING(enroll_id) WHERE t.enroll_id IS NULL),
            (SELECT count(*) FROM truth_meta t
             LEFT JOIN enrollment_meta e USING(enroll_id) WHERE e.enroll_id IS NULL),
            (SELECT count(*) FROM enrollment_meta e JOIN truth_meta t USING(enroll_id)
             WHERE e.source_partition<>t.source_partition),
            (SELECT count(*) FROM enrollment_meta
             WHERE users<>1 OR courses<>1 OR partitions<>1),
            (SELECT count(*) FROM pairs WHERE enrollments<>1)
        """
    ).fetchone()

    temporal_columns = [
        "before_course_start",
        "retained_day_0_to_34",
        "on_or_after_day_35",
        "after_course_end",
        "missing_course_metadata",
    ]
    temporal_row = connection.execute(
        f"""
        WITH logs AS ({logs}),
        course AS (
            SELECT course_id,
                   try_cast(start AS TIMESTAMP) AS course_start,
                   try_cast("end" AS TIMESTAMP) AS course_end
            FROM {courses}
        ),
        parsed AS (
            SELECT try_cast(l.time AS TIMESTAMP) AS event_time,
                   c.course_start, c.course_end,
                   date_diff('day', cast(c.course_start AS DATE),
                                    cast(try_cast(l.time AS TIMESTAMP) AS DATE)) AS course_day
            FROM logs l LEFT JOIN course c USING(course_id)
        )
        SELECT
            count(*) FILTER (WHERE course_day<0),
            count(*) FILTER (WHERE course_day BETWEEN 0 AND 34),
            count(*) FILTER (WHERE course_day>=35),
            count(*) FILTER (WHERE event_time>course_end),
            count(*) FILTER (WHERE course_start IS NULL)
        FROM parsed
        """
    ).fetchone()

    user_columns = [
        "selected_users",
        "missing_metadata",
        "duplicate_metadata_users",
        "missing_gender",
        "missing_education",
        "missing_birth_year",
        "invalid_birth_year",
    ]
    user_row = connection.execute(
        f"""
        WITH logs AS ({logs}), used AS (
            SELECT DISTINCT try_cast(username AS BIGINT) AS user_id FROM logs
        ), profiles AS (
            SELECT try_cast(user_id AS BIGINT) AS user_id,
                   count(*) AS metadata_rows,
                   min(nullif(trim(gender), '')) AS gender,
                   min(nullif(trim(education), '')) AS education,
                   min(nullif(trim(birth), '')) AS birth
            FROM {users} GROUP BY user_id
        )
        SELECT count(*),
               count(*) FILTER (WHERE p.user_id IS NULL),
               count(*) FILTER (WHERE metadata_rows<>1),
               count(*) FILTER (WHERE gender IS NULL),
               count(*) FILTER (WHERE education IS NULL),
               count(*) FILTER (WHERE birth IS NULL),
               count(*) FILTER (
                   WHERE birth IS NOT NULL AND try_cast(birth AS DOUBLE) IS NULL
               )
        FROM used u LEFT JOIN profiles p USING(user_id)
        """
    ).fetchone()

    duration_columns = [
        "used_courses",
        "missing_metadata",
        "duplicate_metadata_courses",
        "duration_under_35_days",
        "min_days",
        "max_days",
        "missing_start",
        "missing_end",
        "missing_course_type",
        "missing_category",
    ]
    duration_row = connection.execute(
        f"""
        WITH logs AS ({logs}), used AS (SELECT DISTINCT course_id FROM logs),
        profiles AS (
            SELECT course_id, count(*) AS metadata_rows,
                   min(nullif(trim(start), '')) AS start,
                   min(nullif(trim("end"), '')) AS "end",
                   min(nullif(trim(course_type), '')) AS course_type,
                   min(nullif(trim(category), '')) AS category
            FROM {courses} GROUP BY course_id
        ), course AS (
            SELECT u.course_id, p.metadata_rows, p.start, p."end",
                   p.course_type, p.category,
                   date_diff('day', try_cast(p.start AS DATE),
                                    try_cast(p."end" AS DATE)) AS duration_days
            FROM used u LEFT JOIN profiles p USING(course_id)
        )
        SELECT count(*), count(*) FILTER (WHERE metadata_rows IS NULL),
               count(*) FILTER (WHERE metadata_rows<>1),
               count(*) FILTER (WHERE duration_days<35),
               min(duration_days), max(duration_days),
               count(*) FILTER (WHERE start IS NULL),
               count(*) FILTER (WHERE "end" IS NULL),
               count(*) FILTER (WHERE course_type IS NULL),
               count(*) FILTER (WHERE category IS NULL)
        FROM course
        """
    ).fetchone()

    duplicate_columns = ["duplicate_groups", "duplicate_rows_beyond_first"]
    duplicate_row = connection.execute(
        f"""
        WITH logs AS ({logs}), duplicate_groups AS (
            SELECT count(*) AS occurrences
            FROM logs
            GROUP BY enroll_id, username, course_id, session_id,
                     action, object, time, source_partition
            HAVING count(*)>1
        )
        SELECT count(*), coalesce(sum(occurrences-1), 0) FROM duplicate_groups
        """
    ).fetchone()

    action_rows = connection.execute(
        f"""
        WITH logs AS ({logs})
        SELECT action, count(*) FROM logs GROUP BY action ORDER BY action
        """
    ).fetchall()

    audit = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "source": validate_source_files(raw_dir),
        "logs": _tuple(log_columns, log_row),
        "logs_by_source_partition": {
            row[0]: dict(zip(("events", "enrollments", "users", "courses"), row[1:], strict=True))
            for row in split_rows
        },
        "labels_by_source_partition": {
            row[0]: dict(
                zip(
                    ("labels", "enrollments", "dropout", "non_dropout", "invalid"),
                    row[1:],
                    strict=True,
                )
            )
            for row in truth_rows
        },
        "enrollment_relations": _tuple(relation_columns, relation_row),
        "temporal": _tuple(temporal_columns, temporal_row),
        "user_profiles": _tuple(user_columns, user_row),
        "course_duration": _tuple(duration_columns, duration_row),
        "duplicates": _tuple(duplicate_columns, duplicate_row),
        "action_counts": {row[0]: row[1] for row in action_rows},
    }
    validate_audit(audit)
    return audit


def validate_audit(audit: dict[str, Any]) -> None:
    errors: list[str] = []
    logs = audit["logs"]
    relations = audit["enrollment_relations"]
    temporal = audit["temporal"]
    users = audit["user_profiles"]
    duration = audit["course_duration"]
    if logs["events"] != DATASET_CONTRACT.raw_events:
        errors.append(f"Expected {DATASET_CONTRACT.raw_events} events, got {logs['events']}")
    if logs["enrollments"] != DATASET_CONTRACT.enrollments:
        errors.append(
            f"Expected {DATASET_CONTRACT.enrollments} enrollments, got {logs['enrollments']}"
        )
    if logs["users"] != DATASET_CONTRACT.users:
        errors.append(f"Expected {DATASET_CONTRACT.users} users, got {logs['users']}")
    if logs["courses"] != DATASET_CONTRACT.courses:
        errors.append(f"Expected {DATASET_CONTRACT.courses} courses, got {logs['courses']}")
    missing_required = {
        field: logs[f"missing_{field}"]
        for field in ("enroll_id", "user", "course", "session", "action", "time")
        if logs[f"missing_{field}"] != 0
    }
    if missing_required:
        errors.append(f"Required event fields contain missing values: {missing_required}")
    if logs["invalid_enroll_id"] != 0 or logs["invalid_user_id"] != 0:
        errors.append("Invalid numeric enrollment or user IDs were found")
    if logs["unknown_action_count"] != 0 or logs["invalid_timestamp"] != 0:
        errors.append("Unknown actions or invalid timestamps were found")
    if any(relations.values()):
        errors.append(f"Enrollment relation invariant failed: {relations}")
    if temporal["retained_day_0_to_34"] != DATASET_CONTRACT.retained_events_35d:
        errors.append("Unexpected retained event count for course days 0-34")
    if temporal["before_course_start"] != 0 or temporal["missing_course_metadata"] != 0:
        errors.append(f"Temporal invariant failed: {temporal}")
    if users["selected_users"] != DATASET_CONTRACT.users:
        errors.append(f"Unexpected user profile count: {users['selected_users']}")
    if users["missing_metadata"] != 0 or users["duplicate_metadata_users"] != 0:
        errors.append(f"User metadata invariant failed: {users}")
    if duration["used_courses"] != DATASET_CONTRACT.courses:
        errors.append(f"Unexpected course profile count: {duration['used_courses']}")
    if duration["missing_metadata"] != 0 or duration["duplicate_metadata_courses"] != 0:
        errors.append(f"Course metadata invariant failed: {duration}")
    if duration["missing_start"] != 0 or duration["missing_end"] != 0:
        errors.append(f"Course date invariant failed: {duration}")
    if duration["duration_under_35_days"] != 0:
        errors.append("A used course is shorter than the 35-day observation window")
    invalid_labels = sum(item["invalid"] for item in audit["labels_by_source_partition"].values())
    label_rows = sum(item["labels"] for item in audit["labels_by_source_partition"].values())
    label_enrollments = sum(
        item["enrollments"] for item in audit["labels_by_source_partition"].values()
    )
    if label_rows != DATASET_CONTRACT.enrollments or label_rows != label_enrollments:
        errors.append(
            f"Expected one label per enrollment, got {label_rows} rows for "
            f"{label_enrollments} enrollment IDs"
        )
    if invalid_labels:
        errors.append(f"Found {invalid_labels} invalid labels")
    if errors:
        raise DataAuditError("\n".join(errors))


def audit_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Any]:
    raw_dir = require_project_path(raw_dir)
    output_dir = require_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    try:
        audit = run_audit(connection, raw_dir)
    finally:
        connection.close()
    write_json_atomic(output_dir / "audit.json", audit)
    return audit
