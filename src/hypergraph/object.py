"""Learning-object hyperedge memberships."""
from __future__ import annotations

from pathlib import Path

from data.cache import sql_path
from data.schema import ACTION_GROUPS


OBJECT_TYPES = ("video", "assignment", "forum")


def _sql_values(values: tuple[str, ...]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


def object_type_expression() -> str:
    """Map event actions to the three enabled object families."""

    clauses = [
        f"WHEN action IN ({_sql_values(ACTION_GROUPS[name])}) THEN '{name}'"
        for name in OBJECT_TYPES
    ]
    return "CASE " + " ".join(clauses) + " END"


def object_membership_query(events_path: Path) -> str:
    """Return unique node-to-object incidences; missing objects are excluded."""

    object_type = object_type_expression()
    enabled_actions = tuple(
        action for name in OBJECT_TYPES for action in ACTION_GROUPS[name]
    )
    return f"""
        SELECT DISTINCT
            'object'::VARCHAR AS family,
            node_id,
            course_id,
            object_id,
            {object_type}::VARCHAR AS object_type
        FROM read_parquet('{sql_path(events_path)}')
        WHERE object_id IS NOT NULL
          AND action IN ({_sql_values(enabled_actions)})
    """
