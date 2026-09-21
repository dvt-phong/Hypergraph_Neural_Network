# Kiểm tra dữ liệu XuetangX gốc và tạo bốn bảng Parquet chuẩn của Phase 1.
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import duckdb

from config import (
    ACTIONS,
    DATASET_CONTRACT,
    OBSERVATION_DAYS,
    SOURCE_PARTITIONS,
    SOURCE_SCHEMAS,
)
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, require_project_path


PARQUET_ARTIFACTS = (
    "nodes.parquet",
    "events_35d.parquet",
    "users.parquet",
    "courses.parquet",
)


# Mục đích: Phân biệt lỗi contract dữ liệu với lỗi chạy Python thông thường.
# Đầu vào: Thông báo mô tả invariant bị vi phạm.
# Đầu ra: Exception làm pipeline dừng ngay.
class DataValidationError(RuntimeError):
    pass


# Mục đích: Liệt kê đầy đủ các file CSV đầu vào theo thứ tự trong contract.
# Đầu vào: Thư mục raw data.
# Đầu ra: Danh sách sáu Path trỏ tới các file CSV bắt buộc.
def source_paths(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    return [raw_dir / name for name in SOURCE_SCHEMAS]


# Mục đích: Kiểm tra file nguồn tồn tại và có đúng header đã khóa.
# Đầu vào: Thư mục chứa sáu file CSV XuetangX.
# Đầu ra: Dictionary kích thước và danh sách cột của từng file.
# Lưu ý: Ném DataValidationError trước khi xử lý dữ liệu nếu có file/header sai.
def validate_source_files(raw_dir: Path = RAW_DATA_DIR) -> dict[str, dict[str, Any]]:
    raw_dir = require_project_path(raw_dir)
    result: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for filename, expected_header in SOURCE_SCHEMAS.items():
        path = raw_dir / filename
        if not path.is_file():
            errors.append(f"Missing source file: {path}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            actual_header = tuple(next(csv.reader(source)))
        if actual_header != expected_header:
            errors.append(
                f"Unexpected header in {filename}: expected {expected_header}, "
                f"got {actual_header}"
            )
        result[filename] = {
            "size_bytes": path.stat().st_size,
            "columns": list(actual_header),
        }
    if errors:
        raise DataValidationError("\n".join(errors))
    return result


# Mục đích: Escape một đường dẫn để dùng an toàn trong SQL nội bộ.
# Đầu vào: Path cần chèn vào câu SQL.
# Đầu ra: Chuỗi đường dẫn tuyệt đối dạng POSIX.
def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


# Mục đích: Tạo biểu thức DuckDB đọc CSV với mọi cột ban đầu là chuỗi.
# Đầu vào: Đường dẫn CSV.
# Đầu ra: Chuỗi biểu thức read_csv_auto(...).
# Lưu ý: Ép kiểu được thực hiện rõ ràng ở query sau để phát hiện dữ liệu lỗi.
def _csv_source(path: Path) -> str:
    return (
        f"read_csv_auto('{_sql_path(path)}', header=true, "
        "all_varchar=true, nullstr='')"
    )


# Mục đích: Ghép train_log và test_log thành một nguồn event thống nhất.
# Đầu vào: Thư mục raw data.
# Đầu ra: Chuỗi SQL UNION ALL có thêm cột source_partition.
# Lưu ý: Đây là chuỗi SQL nhiều dòng, không phải comment khối.
def _logs_sql(raw_dir: Path) -> str:
    train = _csv_source(raw_dir / "train_log.csv")
    test = _csv_source(raw_dir / "test_log.csv")
    return f"""
        SELECT *, 'train' AS source_partition FROM {train}
        UNION ALL
        SELECT *, 'test' AS source_partition FROM {test}
    """


# Mục đích: Ghép train_truth và test_truth thành một nguồn label thống nhất.
# Đầu vào: Thư mục raw data.
# Đầu ra: Chuỗi SQL UNION ALL có thêm cột source_partition.
def _truths_sql(raw_dir: Path) -> str:
    train = _csv_source(raw_dir / "train_truth.csv")
    test = _csv_source(raw_dir / "test_truth.csv")
    return f"""
        SELECT *, 'train' AS source_partition FROM {train}
        UNION ALL
        SELECT *, 'test' AS source_partition FROM {test}
    """


# Mục đích: Ghi kết quả query thành Parquet theo cách atomic.
# Đầu vào: Kết nối DuckDB, câu SELECT và file đích.
# Đầu ra: Không trả dữ liệu; tạo file Parquet hoàn chỉnh.
# Lưu ý: File .part chỉ được đổi tên sau khi DuckDB ghi thành công.
def _write_parquet(
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


# Mục đích: Kiểm tra quan hệ raw đủ rõ ràng để tạo node và label duy nhất.
# Đầu vào: Kết nối DuckDB và thư mục CSV gốc.
# Đầu ra: Không trả dữ liệu khi mọi invariant hợp lệ.
# Lưu ý: Chưa tạo artifact; mọi lỗi đều làm pipeline dừng bằng DataValidationError.
def _validate_raw_relations(
    connection: duckdb.DuckDBPyConnection,
    raw_dir: Path,
) -> None:
    actions = ", ".join("'" + action.replace("'", "''") + "'" for action in ACTIONS)
    row = connection.execute(
        f"""
        WITH logs AS ({_logs_sql(raw_dir)}),
        truths AS ({_truths_sql(raw_dir)}),
        log_meta AS (
            SELECT enroll_id, source_partition,
                   count(DISTINCT username) AS users,
                   count(DISTINCT course_id) AS courses
            FROM logs GROUP BY enroll_id, source_partition
        ),
        truth_meta AS (
            SELECT enroll_id, source_partition,
                   count(*) AS rows,
                   count(DISTINCT truth) AS labels
            FROM truths GROUP BY enroll_id, source_partition
        )
        SELECT
            (SELECT count(*) FROM logs),
            (SELECT count(DISTINCT enroll_id) FROM logs),
            (SELECT count(*) FROM logs
             WHERE enroll_id IS NULL OR trim(enroll_id)=''
                OR username IS NULL OR trim(username)=''
                OR course_id IS NULL OR trim(course_id)=''
                OR session_id IS NULL OR trim(session_id)=''
                OR action IS NULL OR trim(action)=''
                OR time IS NULL OR trim(time)=''),
            (SELECT count(*) FROM logs
             WHERE try_cast(enroll_id AS BIGINT) IS NULL
                OR try_cast(username AS BIGINT) IS NULL
                OR try_cast(time AS TIMESTAMP) IS NULL),
            (SELECT count(*) FROM logs WHERE action NOT IN ({actions})),
            (SELECT count(*) FROM truths),
            (SELECT count(DISTINCT enroll_id) FROM truths),
            (SELECT count(*) FROM truths
             WHERE try_cast(enroll_id AS BIGINT) IS NULL
                OR try_cast(truth AS INTEGER) NOT IN (0, 1)),
            (SELECT count(*) FROM log_meta WHERE users<>1 OR courses<>1),
            (SELECT count(*) FROM truth_meta WHERE rows<>1 OR labels<>1),
            (SELECT count(*) FROM log_meta l
             LEFT JOIN truth_meta t USING(enroll_id, source_partition)
             WHERE t.enroll_id IS NULL),
            (SELECT count(*) FROM truth_meta t
             LEFT JOIN log_meta l USING(enroll_id, source_partition)
             WHERE l.enroll_id IS NULL)
        """
    ).fetchone()
    names = (
        "raw_events",
        "log_enrollments",
        "missing_required_log_fields",
        "invalid_log_types",
        "unknown_actions",
        "truth_rows",
        "truth_enrollments",
        "invalid_truth_rows",
        "conflicting_enrollment_metadata",
        "duplicate_or_conflicting_truth",
        "logs_without_truth",
        "truth_without_logs",
    )
    values = dict(zip(names, map(int, row), strict=True))
    expected = DATASET_CONTRACT
    errors: list[str] = []
    if values["raw_events"] != expected.raw_events:
        errors.append(
            f"expected {expected.raw_events} raw events, got {values['raw_events']}"
        )
    for key in ("log_enrollments", "truth_rows", "truth_enrollments"):
        if values[key] != expected.enrollments:
            errors.append(
                f"expected {expected.enrollments} for {key}, got {values[key]}"
            )
    for key in names[2:5] + names[7:]:
        if values[key] != 0:
            errors.append(f"{key}={values[key]}")
    if errors:
        raise DataValidationError("Invalid raw XuetangX data: " + "; ".join(errors))


# Mục đích: Kiểm tra bốn bảng canonical sau khi đã ghi ra đĩa.
# Đầu vào: Kết nối DuckDB và thư mục processed data.
# Đầu ra: Dictionary số dòng của nodes, events, users và courses.
# Lưu ý: So sánh trực tiếp với DATASET_CONTRACT và kiểm tra ngày quan sát 0-34.
def _validate_outputs(
    connection: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> dict[str, int]:
    paths = {name: _sql_path(output_dir / name) for name in PARQUET_ARTIFACTS}
    expected = DATASET_CONTRACT
    allowed_sources = ", ".join(f"'{value}'" for value in SOURCE_PARTITIONS)
    allowed_actions = ", ".join(f"'{value}'" for value in ACTIONS)

    node_row = connection.execute(
        f"""
        SELECT count(*), count(DISTINCT node_id), count(DISTINCT enroll_id),
               min(node_id), max(node_id),
               count(*) FILTER (WHERE node_id IS NULL OR enroll_id IS NULL
                                  OR user_id IS NULL OR course_id IS NULL
                                  OR source_partition IS NULL OR label IS NULL),
               count(*) FILTER (WHERE source_partition NOT IN ({allowed_sources})),
               count(*) FILTER (WHERE label NOT IN (0, 1))
        FROM read_parquet('{paths['nodes.parquet']}')
        """
    ).fetchone()
    expected_nodes = (
        expected.enrollments,
        expected.enrollments,
        expected.enrollments,
        0,
        expected.enrollments - 1,
        0,
        0,
        0,
    )
    if tuple(map(int, node_row)) != expected_nodes:
        raise DataValidationError(
            f"nodes.parquet invariant failed: expected {expected_nodes}, got {node_row}"
        )

    user_row = connection.execute(
        f"""
        SELECT count(*), count(DISTINCT user_id),
               count(*) FILTER (WHERE user_id IS NULL)
        FROM read_parquet('{paths['users.parquet']}')
        """
    ).fetchone()
    if tuple(map(int, user_row)) != (expected.users, expected.users, 0):
        raise DataValidationError(f"users.parquet invariant failed: {user_row}")

    course_row = connection.execute(
        f"""
        SELECT count(*), count(DISTINCT course_id),
               count(*) FILTER (WHERE course_id IS NULL OR course_start IS NULL
                                  OR course_end IS NULL),
               count(*) FILTER (
                   WHERE date_diff('day', course_start::DATE, course_end::DATE)
                         < {OBSERVATION_DAYS}
               )
        FROM read_parquet('{paths['courses.parquet']}')
        """
    ).fetchone()
    if tuple(map(int, course_row)) != (expected.courses, expected.courses, 0, 0):
        raise DataValidationError(f"courses.parquet invariant failed: {course_row}")

    event_row = connection.execute(
        f"""
        SELECT count(*), min(course_day), max(course_day),
               count(*) FILTER (WHERE node_id IS NULL OR enroll_id IS NULL
                                  OR user_id IS NULL OR course_id IS NULL
                                  OR session_id IS NULL OR action IS NULL
                                  OR event_time IS NULL OR course_day IS NULL),
               count(*) FILTER (WHERE source_partition NOT IN ({allowed_sources})),
               count(*) FILTER (WHERE action NOT IN ({allowed_actions}))
        FROM read_parquet('{paths['events_35d.parquet']}')
        """
    ).fetchone()
    expected_events = (expected.retained_events_35d, 0, OBSERVATION_DAYS - 1, 0, 0, 0)
    if tuple(map(int, event_row)) != expected_events:
        raise DataValidationError(
            "events_35d.parquet invariant failed: "
            f"expected {expected_events}, got {event_row}"
        )

    return {
        "nodes.parquet": int(node_row[0]),
        "events_35d.parquet": int(event_row[0]),
        "users.parquet": int(user_row[0]),
        "courses.parquet": int(course_row[0]),
    }


# Mục đích: Chạy trọn Phase 1 từ CSV gốc đến bốn bảng canonical đã kiểm tra.
# Đầu vào: Thư mục raw data và thư mục output processed.
# Đầu ra: Report gồm dataset, observation window, file nguồn và artifact đã tạo.
# Lưu ý: Hàm luôn rebuild; thứ tự node_id ổn định theo enroll_id/source partition.
def prepare_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Any]:
    raw_dir = require_project_path(raw_dir)
    output_dir = require_project_path(output_dir)
    source_files = validate_source_files(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    try:
        _validate_raw_relations(connection, raw_dir)
        logs = _logs_sql(raw_dir)
        truths = _truths_sql(raw_dir)
        raw_courses = _csv_source(raw_dir / "course_info.csv")
        raw_users = _csv_source(raw_dir / "user_info.csv")

        nodes_query = f"""
            WITH logs AS ({logs}), truths AS ({truths}),
            enrollment_meta AS (
                SELECT try_cast(enroll_id AS BIGINT) AS enroll_id,
                       try_cast(min(username) AS BIGINT) AS user_id,
                       min(course_id) AS course_id,
                       source_partition
                FROM logs GROUP BY enroll_id, source_partition
            ),
            labels AS (
                SELECT try_cast(enroll_id AS BIGINT) AS enroll_id,
                       source_partition,
                       try_cast(min(truth) AS TINYINT) AS label
                FROM truths GROUP BY enroll_id, source_partition
            )
            SELECT row_number() OVER (
                       ORDER BY e.enroll_id, e.source_partition
                   ) - 1 AS node_id,
                   e.enroll_id, e.user_id, e.course_id,
                   e.source_partition, l.label
            FROM enrollment_meta e
            JOIN labels l USING(enroll_id, source_partition)
            ORDER BY e.enroll_id, e.source_partition
        """
        _write_parquet(connection, nodes_query, output_dir / "nodes.parquet")

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
        _write_parquet(connection, users_query, output_dir / "users.parquet")

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
        _write_parquet(connection, courses_query, output_dir / "courses.parquet")

        events_query = f"""
            WITH logs AS ({logs}), parsed AS (
                SELECT try_cast(enroll_id AS BIGINT) AS enroll_id,
                       source_partition, session_id, action,
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
                  USING(enroll_id, source_partition)
                JOIN read_parquet('{_sql_path(output_dir / 'courses.parquet')}') c
                  USING(course_id)
            )
            SELECT * FROM canonical
            WHERE course_day BETWEEN 0 AND {OBSERVATION_DAYS - 1}
        """
        _write_parquet(connection, events_query, output_dir / "events_35d.parquet")
        row_counts = _validate_outputs(connection, output_dir)
    finally:
        connection.close()

    return {
        "dataset": DATASET_CONTRACT.dataset,
        "observation_days": OBSERVATION_DAYS,
        "source_files": source_files,
        "artifacts": [
            {
                "path": name,
                "rows": row_counts[name],
                "size_bytes": (output_dir / name).stat().st_size,
            }
            for name in PARQUET_ARTIFACTS
        ],
    }
