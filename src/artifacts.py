# Các hàm nhỏ dùng chung để kiểm tra cache và ghi artifact an toàn.
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb


BUFFER_SIZE = 8 * 1024 * 1024


# Mục đích: Chuyển Path thành đường dẫn tuyệt đối có thể chèn vào câu SQL DuckDB.
# Đầu vào: path là đường dẫn file hoặc thư mục.
# Đầu ra: Chuỗi đường dẫn POSIX; dấu nháy đơn đã được escape.
# Lưu ý: Hàm chỉ chuẩn hóa chuỗi, không kiểm tra file có tồn tại hay không.
def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


# Mục đích: Ghi kết quả một câu query thành Parquet mà không để lại file dở dang.
# Đầu vào: Kết nối DuckDB, câu SELECT và đường dẫn file đích.
# Đầu ra: Không trả dữ liệu; tạo hoặc thay thế file Parquet đích.
# Lưu ý: Dữ liệu được ghi vào file .part rồi mới đổi tên sau khi query thành công.
def copy_parquet_atomic(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    destination: Path,
) -> None:
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    connection.execute(
        f"COPY ({query}) TO '{sql_path(temporary)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)"
    )
    temporary.replace(destination)


# Mục đích: Tính mã SHA-256 để nhận diện chính xác nội dung một file.
# Đầu vào: Đường dẫn tới file cần đọc.
# Đầu ra: Chuỗi hash SHA-256 gồm 64 ký tự hexadecimal.
# Lưu ý: File được đọc theo từng khối 8 MB để không chiếm nhiều RAM.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


# Mục đích: Tạo thông tin nhận diện dùng cho cache và checkpoint.
# Đầu vào: Đường dẫn file và cờ include_hash cho biết có tính SHA-256 hay không.
# Đầu ra: Dictionary gồm tên file, kích thước, thời gian sửa và hash nếu được yêu cầu.
# Lưu ý: Tính hash chính xác hơn nhưng chậm hơn so với chỉ đọc metadata file.
def source_signature(path: Path, *, include_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    signature: dict[str, Any] = {
        "path": path.name,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        signature["sha256"] = sha256(path)
    return signature


# Mục đích: Kiểm tra các file đầu vào có còn giống lúc cache được tạo hay không.
# Đầu vào: Danh sách Path hiện tại và danh sách signature đã lưu trong manifest.
# Đầu ra: True nếu mọi kích thước và mtime đều khớp; ngược lại trả False.
# Lưu ý: Hàm này kiểm tra cache nhanh, không tính lại SHA-256.
def signatures_match(paths: list[Path], recorded: list[dict[str, Any]]) -> bool:
    previous = {item.get("path"): item for item in recorded}
    for path in paths:
        item = previous.get(path.name)
        if item is None:
            return False
        current = source_signature(path, include_hash=False)
        if current["size_bytes"] != item.get("size_bytes"):
            return False
        if current["mtime_ns"] != item.get("mtime_ns"):
            return False
    return True


# Mục đích: Ghi dictionary thành JSON theo cách atomic.
# Đầu vào: Đường dẫn đích và payload có thể serialize thành JSON.
# Đầu ra: Không trả dữ liệu; tạo hoặc thay thế file JSON đích.
# Lưu ý: Dùng file .part để file cũ không bị hỏng nếu quá trình ghi thất bại.
def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


# Mục đích: Lấy thời gian UTC hiện tại để ghi vào report và manifest.
# Đầu vào: Không có.
# Đầu ra: Chuỗi thời gian ISO-8601 có timezone UTC.
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
