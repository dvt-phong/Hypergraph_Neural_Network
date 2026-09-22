# Sinh Course và Object hyperedge candidates từ dữ liệu quan sát được.
from __future__ import annotations

from pathlib import Path

from artifacts import sql_path
from config import ACTION_GROUPS


OBJECT_TYPES = ("video", "assignment", "forum")


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


# Mục đích: Chuyển các giá trị action thành danh sách SQL literal an toàn.
# Đầu vào: Tuple các chuỗi action.
# Đầu ra: Chuỗi dạng 'a', 'b', ... để dùng trong mệnh đề IN.
def _sql_values(values: tuple[str, ...]) -> str:
    escaped_values = []
    for value in values:
        escaped_values.append("'" + value.replace("'", "''") + "'")
    return ", ".join(escaped_values)


# Mục đích: Ánh xạ action sang video, assignment hoặc forum.
# Đầu vào: Không có; đọc ACTION_GROUPS và OBJECT_TYPES.
# Đầu ra: Chuỗi biểu thức SQL CASE.
# Lưu ý: Web-page action không tạo Object hyperedge trong cấu hình chính.
def object_type_expression() -> str:
    clauses = []
    for object_type in OBJECT_TYPES:
        actions = _sql_values(ACTION_GROUPS[object_type])
        clauses.append(f"WHEN action IN ({actions}) THEN '{object_type}'")
    return "CASE " + " ".join(clauses) + " END"


# Mục đích: Tạo unique node-to-object incidence từ event quan sát được.
# Đầu vào: Đường dẫn events_35d.parquet.
# Đầu ra: Chuỗi SQL trả family, node, course-object key và object type.
# Lưu ý: Loại event thiếu object_id và action không thuộc ba family hỗ trợ.
def object_membership_query(events_path: Path) -> str:
    object_type = object_type_expression()
    enabled_actions = []
    for name in OBJECT_TYPES:
        enabled_actions.extend(ACTION_GROUPS[name])
    return f"""
        SELECT DISTINCT
            'object'::VARCHAR AS family,
            node_id,
            course_id,
            object_id,
            {object_type}::VARCHAR AS object_type
        FROM read_parquet('{sql_path(events_path)}')
        WHERE object_id IS NOT NULL
          AND action IN ({_sql_values(tuple(enabled_actions))})
    """
