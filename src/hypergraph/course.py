# Sinh incidence candidate cho family Course.
from __future__ import annotations

from pathlib import Path

from artifacts import sql_path


# Mục đích: Tạo một Course membership cho mỗi enrollment node.
# Đầu vào: Đường dẫn nodes.parquet.
# Đầu ra: Chuỗi SQL gồm family, node_id, course_id và metadata rỗng.
# Lưu ý: Các course singleton sẽ được lọc khi materialize train H0.
def course_membership_query(nodes_path: Path) -> str:
    return f"""
        SELECT
            'course'::VARCHAR AS family,
            node_id,
            course_id,
            NULL::VARCHAR AS object_id,
            NULL::VARCHAR AS object_type
        FROM read_parquet('{sql_path(nodes_path)}')
    """
