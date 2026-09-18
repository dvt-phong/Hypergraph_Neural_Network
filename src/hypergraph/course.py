"""Course-family hyperedge memberships."""
from __future__ import annotations

from pathlib import Path

from data.cache import sql_path


def course_membership_query(nodes_path: Path) -> str:
    """Return one course incidence for every enrollment node."""

    return f"""
        SELECT
            'course'::VARCHAR AS family,
            node_id,
            course_id,
            NULL::VARCHAR AS object_id,
            NULL::VARCHAR AS object_type
        FROM read_parquet('{sql_path(nodes_path)}')
    """
