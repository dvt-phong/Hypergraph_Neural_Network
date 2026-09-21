# Tạo feature nhân khẩu học và bối cảnh course ở mức enrollment node.
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from artifacts import copy_parquet_atomic, sql_path
from config import DATASET_CONTRACT


RAW_CONTEXT_COLUMNS = (
    "gender",
    "education",
    "age_at_course_start",
    "category",
    "course_duration_days",
)


# Mục đích: Sinh query join metadata user/course vào từng node enrollment.
# Đầu vào: Đường dẫn nodes, users và courses Parquet.
# Đầu ra: Chuỗi SQL trả một hàng context cho mỗi node_id.
# Lưu ý: Tuổi ngoài 10-100 và duration âm được đổi thành NULL để xử lý sau.
def _context_query(nodes_path: Path, users_path: Path, courses_path: Path) -> str:
    return f"""
        SELECT n.node_id,
               u.gender,
               u.education,
               CASE
                   WHEN u.birth_year IS NULL THEN NULL
                   WHEN year(c.course_start)-u.birth_year NOT BETWEEN 10 AND 100
                       THEN NULL
                   ELSE year(c.course_start)-u.birth_year
               END::INTEGER AS age_at_course_start,
               c.category,
               CASE
                   WHEN c.course_start IS NULL OR c.course_end IS NULL THEN NULL
                   WHEN date_diff('day', c.course_start, c.course_end)<0 THEN NULL
                   ELSE date_diff('day', c.course_start, c.course_end)
               END::INTEGER AS course_duration_days
        FROM read_parquet('{sql_path(nodes_path)}') n
        LEFT JOIN read_parquet('{sql_path(users_path)}') u USING(user_id)
        LEFT JOIN read_parquet('{sql_path(courses_path)}') c USING(course_id)
        ORDER BY n.node_id
    """


# Mục đích: Kiểm tra bảng context có đủ node và giá trị numeric hợp lệ.
# Đầu vào: Kết nối DuckDB và đường dẫn context_raw.parquet.
# Đầu ra: Dictionary thống kê missing/invalid của từng nhóm context.
# Lưu ý: Missing được cho phép; invalid age hoặc duration làm pipeline dừng.
def validate_raw_context(
    connection: duckdb.DuckDBPyConnection, context_path: Path
) -> dict[str, Any]:
    source = f"read_parquet('{sql_path(context_path)}')"
    row = connection.execute(
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT node_id) AS node_ids,
               min(node_id) AS min_node_id,
               max(node_id) AS max_node_id,
               count(*) FILTER (WHERE gender IS NULL) AS missing_gender,
               count(*) FILTER (WHERE education IS NULL) AS missing_education,
               count(*) FILTER (WHERE age_at_course_start IS NULL) AS missing_age,
               count(*) FILTER (
                   WHERE age_at_course_start IS NOT NULL
                     AND age_at_course_start NOT BETWEEN 10 AND 100
               ) AS invalid_age,
               count(*) FILTER (WHERE category IS NULL) AS missing_category,
               count(*) FILTER (WHERE course_duration_days IS NULL)
                   AS missing_duration,
               count(*) FILTER (WHERE course_duration_days<0)
                   AS invalid_duration
        FROM {source}
        """
    ).fetchone()
    keys = (
        "rows",
        "distinct_node_ids",
        "min_node_id",
        "max_node_id",
        "missing_gender",
        "missing_education",
        "missing_age",
        "invalid_age",
        "missing_category",
        "missing_duration",
        "invalid_duration",
    )
    audit = {key: int(value) for key, value in zip(keys, row, strict=True)}
    expected = {
        "rows": DATASET_CONTRACT.enrollments,
        "distinct_node_ids": DATASET_CONTRACT.enrollments,
        "min_node_id": 0,
        "max_node_id": DATASET_CONTRACT.enrollments - 1,
        "invalid_age": 0,
        "invalid_duration": 0,
    }
    for key, value in expected.items():
        if audit[key] != value:
            raise RuntimeError(
                f"Raw context invariant failed for {key}: "
                f"expected {value}, got {audit[key]}"
            )
    return audit


# Mục đích: Join metadata và ghi bảng context thô cho mọi enrollment.
# Đầu vào: Kết nối DuckDB, ba bảng nguồn và đường dẫn file đích.
# Đầu ra: Audit dictionary của bảng context vừa tạo.
# Lưu ý: Chưa one-hot hoặc standardize ở bước này.
def build_raw_context(
    connection: duckdb.DuckDBPyConnection,
    nodes_path: Path,
    users_path: Path,
    courses_path: Path,
    destination: Path,
) -> dict[str, Any]:
    copy_parquet_atomic(
        connection,
        _context_query(nodes_path, users_path, courses_path),
        destination,
    )
    return validate_raw_context(connection, destination)
