"""Thống kê TOÀN BỘ CSV XuetangX, xuất bảng Markdown và JSON.

Cài một lần: .venv/Scripts/python.exe -m pip install -r scripts/requirements-data.txt
Chạy: .venv/Scripts/python.exe scripts/profile_xuetangx.py
DuckDB xử lý dữ liệu lớn và dùng ổ đĩa tạm khi cần; không cần GPU/pandas.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/xuetangx"
REPORT = ROOT / "outputs/reports/xuetangx"
MEANINGS = {
    "enroll_id": "Mã lượt đăng ký (người học, khóa học); nối log với truth.",
    "username": "Mã người học trong log; nối với user_info.user_id.",
    "session_id": "Mã phiên hoạt động.",
    "action": "Loại hành động học tập; phân bố xem bên dưới.",
    "object": "Đối tượng của hành động; giữ dạng chuỗi để không mất thông tin.",
    "time": "Thời điểm hành động; nguồn không chỉ rõ múi giờ.",
    "truth": "Nhãn: 1 = bỏ học, 0 = không bỏ học.",
    "user_id": "Mã người học trong bảng hồ sơ.",
    "gender": "Giới tính được ghi trong hồ sơ.",
    "education": "Trình độ học vấn được ghi trong hồ sơ.",
    "birth": "Năm sinh; số thực trong CSV, không phải tuổi.",
    "id": "Mã nội bộ khóa học; dữ liệu thực tế có cả giá trị không thuần số.",
    "start": "Thời điểm bắt đầu khóa học.",
    "end": "Thời điểm kết thúc khóa học.",
    "course_type": "Hình thức học: 0 = instructor-paced, 1 = self-paced.",
    "category": "Danh mục khóa học; giữ nguyên mã/chuỗi do nguồn cung cấp.",
}


def identifier(value):
    """Quote tên cột cho SQL (ví dụ cột end là từ khóa)."""
    return '"' + value.replace('"', '""') + '"'


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect()
    db.execute("SET memory_limit = '1GB'")
    db.execute("SET threads = 2")
    db.execute("SET temp_directory = ?", [str(REPORT / ".duckdb_tmp")])
    results = []
    files = [DATA / f"{split}_{kind}.csv" for split in ("train", "test")
             for kind in ("log", "truth")]
    files += [DATA / "user_info.csv", DATA / "course_info.csv"]
    for path in files:
        print(f"Profiling: {path.name}", flush=True)
        # Đọc nguyên chuỗi: không biến mã ID thành số hoặc coi 'NA' là thiếu.
        db.execute("CREATE OR REPLACE TABLE current_data AS SELECT * FROM "
                   "read_csv(?, header=true, all_varchar=true, nullstr='', "
                   "sample_size=-1, strict_mode=true)", [str(path)])
        rows = db.execute("SELECT count(*) FROM current_data").fetchone()[0]
        columns = [row[0] for row in db.execute("DESCRIBE current_data").fetchall()]
        fields = []
        for name in columns:
            print(f"  Field: {name}", flush=True)
            col = identifier(name)
            missing, distinct, numeric, timestamps, low, high = db.execute(f"""
                SELECT count(*) FILTER (WHERE {col} IS NULL),
                       count(DISTINCT {col}),
                       count(try_cast({col} AS DOUBLE)),
                       count(try_cast({col} AS TIMESTAMP)),
                       min({col}), max({col}) FROM current_data
            """).fetchone()
            present = rows - missing
            dtype = "empty" if not present else "string"
            if present and numeric == present:
                dtype = "number"
                low, high = db.execute(f"SELECT min(cast({col} AS DOUBLE)), "
                                      f"max(cast({col} AS DOUBLE)) FROM current_data").fetchone()
            elif present and timestamps == present:
                dtype = "timestamp"
            top = db.execute(f"SELECT {col}, count(*) AS n FROM current_data "
                             f"WHERE {col} IS NOT NULL GROUP BY {col} "
                             f"ORDER BY n DESC, {col} LIMIT 5").fetchall()
            meaning = MEANINGS.get(name, "Chưa có mô tả từ nguồn.")
            if name == "course_id":
                meaning = ("Mã chuỗi khóa học; dùng trong tracking log gốc."
                           if path.name == "course_info.csv"
                           else "Mã chuỗi khóa học trong file thực tế; nối với course_info.course_id.")
            fields.append(dict(field=name, inferred_type=dtype, missing=missing,
                               missing_percent=round(100 * missing / rows, 4) if rows else 0,
                               distinct=distinct, minimum=low, maximum=high,
                               top5=top, meaning=meaning))
        results.append(dict(file=path.name, rows=rows, bytes=path.stat().st_size, fields=fields))
        print(f"  {rows:,} rows; {len(fields)} fields", flush=True)
    db.close()
    write_report(results)


def write_report(results):
    """Xuất cùng một kết quả thống kê ra JSON và bảng dễ đọc."""
    report = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                  source="http://moocdata.cn/data/user-activity", files=results)
    (REPORT / "statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# XuetangX — thống kê dữ liệu tải thực tế", "",
             f"Thời điểm: {report['generated_utc']}", "",
             "Nguồn: [MoocData – User Activity](http://moocdata.cn/data/user-activity).",
             "Thống kê toàn bộ 6 CSV, không lấy mẫu. Thiếu = ô rỗng; distinct không tính ô thiếu.",
             "Kiểu dữ liệu được suy ra từ toàn bộ giá trị; ID dù chứa chữ số vẫn là mã định danh.",
             "Min/max của chuỗi theo thứ tự từ điển; không mang nghĩa độ lớn.", "",
             "| File | Số dòng | Số field | Dung lượng (MiB) |", "|---|---:|---:|---:|"]
    for item in results:
        lines.append(f"| {item['file']} | {item['rows']:,} | {len(item['fields'])} | {item['bytes']/1024**2:.2f} |")
    for item in results:
        lines += ["", f"## {item['file']}", "",
                  "| Field | Kiểu suy ra | Thiếu | Thiếu (%) | Phân biệt | Min | Max | Ý nghĩa |",
                  "|---|---|---:|---:|---:|---|---|---|"]
        for f in item["fields"]:
            values = [f['field'], f['inferred_type'], f"{f['missing']:,}", f['missing_percent'],
                      f"{f['distinct']:,}", f['minimum'], f['maximum'], f['meaning']]
            lines.append("| " + " | ".join(map(cell, values)) + " |")
        lines += ["", "Giá trị phổ biến nhất (tối đa 5; số lần xuất hiện):", "",
                  "| Field | Giá trị và tần suất |", "|---|---|"]
        for f in item['fields']:
            lines.append(f"| {f['field']} | " + cell("; ".join(f"{v}: {n:,}" for v, n in f['top5'])) + " |")
    lines += ["", "## Lưu ý sử dụng", "",
              "- Khóa nối thực tế: log.username = user_info.user_id; log.course_id = course_info.course_id; log.enroll_id = truth.enroll_id.",
              "- Trang nguồn mô tả course_id của bộ dự đoán là mã số, nhưng CSV tải thực tế dùng mã chuỗi. Bảng này mô tả CSV thực tế.",
              "- birth là năm sinh tự khai; kiểm tra min/max và lọc giá trị bất hợp lý trước khi tính tuổi.",
              "- Hồ sơ người học và danh mục khóa học có thể bao phủ nhiều đối tượng hơn tập dự đoán.",
              "- Giữ nguyên train/test khi tải; script này không gộp, lọc hay tạo feature.",
              "- SIG-Net gốc công bố KDD Cup 2015 và NAVER, không kèm loader XuetangX.",
              "- MST-GCN có loader đọc bốn file train/test log/truth. Chưa xác nhận bản dữ liệu hoặc split tác giả thực sự chạy bằng checksum.",
              "- MST-GCN hiện hard-code data/xuetangx; project này lưu ở data/raw/xuetangx theo configs/xuetangx.yaml.",
              "- Loader MST-GCN được clone suy ra ngày bắt đầu từ log và gộp train/test; cần kiểm tra protocol trước khi tái lập kết quả.", ""]
    (REPORT / "statistics.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Reports: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
