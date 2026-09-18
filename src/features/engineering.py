"""Aggregate canonical events into enrollment-level raw features."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from data.cache import copy_parquet_atomic, sql_path
from data.schema import (
    ACTIONS,
    DATASET_CONTRACT,
    EARLY_OBSERVATION_DAYS,
    OBSERVATION_DAYS,
    base_feature_columns,
)


ABLATION_FEATURES = (
    "event_count",
    "active_days",
    "active_span_days",
    "first_active_day",
    "last_active_day",
    "days_since_last_activity",
    "active_day_ratio",
    "has_activity",
)


def _aggregate_query(
    nodes_path: Path,
    events_path: Path,
    observation_days: int,
) -> str:
    day_aggregates = [
        f"count(*) FILTER (WHERE course_day={day}) AS day_{day:02d}"
        for day in range(observation_days)
    ]
    action_aggregates = [
        f"count(*) FILTER (WHERE action='{action}') AS action_{action}"
        for action in ACTIONS
    ]
    aggregates = ",\n                   ".join(day_aggregates + action_aggregates)
    base_counts = ",\n               ".join(
        f"coalesce(a.{column}, 0)::BIGINT AS {column}"
        for column in base_feature_columns(observation_days)
    )
    return f"""
        WITH aggregated AS (
            SELECT node_id,
                   {aggregates},
                   count(DISTINCT session_id) AS session_count,
                   count(DISTINCT (course_id, object_id)) FILTER (
                       WHERE object_id IS NOT NULL
                   ) AS distinct_observed_objects,
                   count(*) AS event_count,
                   count(DISTINCT course_day) AS active_days,
                   min(course_day) AS first_active_day,
                   max(course_day) AS last_active_day
            FROM read_parquet('{sql_path(events_path)}')
            WHERE course_day<{observation_days}
            GROUP BY node_id
        )
        SELECT n.node_id,
               {base_counts},
               coalesce(a.event_count, 0)::BIGINT AS event_count,
               coalesce(a.active_days, 0)::BIGINT AS active_days,
               CASE WHEN a.event_count IS NULL THEN 0
                    ELSE a.last_active_day-a.first_active_day+1 END::BIGINT
                    AS active_span_days,
               coalesce(a.first_active_day, 0)::BIGINT AS first_active_day,
               coalesce(a.last_active_day, 0)::BIGINT AS last_active_day,
               CASE WHEN a.event_count IS NULL THEN {observation_days}
                    ELSE {observation_days - 1}-a.last_active_day END::BIGINT
                    AS days_since_last_activity,
               coalesce(a.active_days, 0)::DOUBLE/{observation_days}
                    AS active_day_ratio,
               CASE WHEN a.event_count IS NULL THEN 0 ELSE 1 END::TINYINT
                    AS has_activity
        FROM read_parquet('{sql_path(nodes_path)}') n
        LEFT JOIN aggregated a USING(node_id)
        ORDER BY n.node_id
    """


def validate_raw_features(
    connection: duckdb.DuckDBPyConnection,
    features_path: Path,
    *,
    observation_days: int,
    expected_events: int,
) -> dict[str, Any]:
    source = f"read_parquet('{sql_path(features_path)}')"
    day_sum = " + ".join(f"day_{day:02d}" for day in range(observation_days))
    action_sum = " + ".join(f"action_{action}" for action in ACTIONS)
    base_columns = base_feature_columns(observation_days)
    missing_or_negative = " OR ".join(
        f"{column} IS NULL OR {column}<0" for column in base_columns
    )
    row = connection.execute(
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT node_id) AS node_ids,
               min(node_id) AS min_node_id,
               max(node_id) AS max_node_id,
               sum(event_count) AS events,
               count(*) FILTER (WHERE event_count<>{day_sum}) AS bad_daily_sum,
               count(*) FILTER (WHERE event_count<>{action_sum}) AS bad_action_sum,
               count(*) FILTER (WHERE {missing_or_negative}) AS bad_base_values,
               count(*) FILTER (
                   WHERE session_count>event_count
                      OR distinct_observed_objects>event_count
                      OR active_days>{observation_days}
                      OR has_activity NOT IN (0, 1)
                      OR (event_count=0 AND has_activity<>0)
                      OR (event_count>0 AND has_activity<>1)
               ) AS bad_summary_values,
               count(*) FILTER (WHERE event_count=0) AS zero_activity_nodes
        FROM {source}
        """
    ).fetchone()
    audit = {
        "rows": int(row[0]),
        "distinct_node_ids": int(row[1]),
        "min_node_id": int(row[2]),
        "max_node_id": int(row[3]),
        "events": int(row[4]),
        "bad_daily_sum": int(row[5]),
        "bad_action_sum": int(row[6]),
        "bad_base_values": int(row[7]),
        "bad_summary_values": int(row[8]),
        "zero_activity_nodes": int(row[9]),
    }
    expected = DATASET_CONTRACT
    expected_values = {
        "rows": expected.enrollments,
        "distinct_node_ids": expected.enrollments,
        "min_node_id": 0,
        "max_node_id": expected.enrollments - 1,
        "events": expected_events,
        "bad_daily_sum": 0,
        "bad_action_sum": 0,
        "bad_base_values": 0,
        "bad_summary_values": 0,
    }
    for key, value in expected_values.items():
        if audit[key] != value:
            raise RuntimeError(
                f"Raw feature invariant failed for {key}: "
                f"expected {value}, got {audit[key]}"
            )
    return audit


def build_raw_features(
    connection: duckdb.DuckDBPyConnection,
    nodes_path: Path,
    events_path: Path,
    destination: Path,
    *,
    observation_days: int = OBSERVATION_DAYS,
) -> dict[str, Any]:
    """Build windowed base features and eight activity-ablation fields."""

    if observation_days not in EARLY_OBSERVATION_DAYS:
        raise ValueError(
            f"observation_days must be one of {EARLY_OBSERVATION_DAYS}, "
            f"got {observation_days}"
        )
    expected_events = (
        DATASET_CONTRACT.retained_events_35d
        if observation_days == OBSERVATION_DAYS
        else int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet('{sql_path(events_path)}') "
                "WHERE course_day<?",
                [observation_days],
            ).fetchone()[0]
        )
    )

    copy_parquet_atomic(
        connection,
        _aggregate_query(nodes_path, events_path, observation_days),
        destination,
    )
    return validate_raw_features(
        connection,
        destination,
        observation_days=observation_days,
        expected_events=expected_events,
    )
