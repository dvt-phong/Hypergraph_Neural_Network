# Sinh incidence candidate cho learning-object hyperedges.
from __future__ import annotations

from pathlib import Path

from artifacts import sql_path
from config import ACTION_GROUPS


OBJECT_TYPES = ("video", "assignment", "forum")


# Mục đích: Chuyển tuple chuỗi thành danh sách SQL literal an toàn.
# Đầu vào: Tuple action hoặc category strings.
# Đầu ra: Chuỗi dạng 'a', 'b', ... để dùng trong mệnh đề IN.
def _sql_values(values: tuple[str, ...]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


# Mục đích: Ánh xạ event action sang video, assignment hoặc forum.
# Đầu vào: Không có; đọc ACTION_GROUPS và OBJECT_TYPES.
# Đầu ra: Chuỗi biểu thức SQL CASE.
# Lưu ý: Web-page action không tạo Object hyperedge trong cấu hình chính.
def object_type_expression() -> str:
    clauses = [
        f"WHEN action IN ({_sql_values(ACTION_GROUPS[name])}) THEN '{name}'"
        for name in OBJECT_TYPES
    ]
    return "CASE " + " ".join(clauses) + " END"


# Mục đích: Tạo unique node-to-object incidence từ event quan sát được.
# Đầu vào: Đường dẫn events_35d.parquet.
# Đầu ra: Chuỗi SQL trả family, node, course-object key và object type.
# Lưu ý: Loại event thiếu object_id và action không thuộc ba family hỗ trợ.
def object_membership_query(events_path: Path) -> str:
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
