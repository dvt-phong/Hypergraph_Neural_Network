r"""Stream XuetangX full-log JSON files into gzip-compressed CSV files.

Usage:
    .\.venv\Scripts\python.exe scripts\convert_xuetangx_full.py

Input JSON structure:
    [course_id, {user_id: {session_id: [[action, time], ...]}}]

The converter never loads a complete JSON file, course, or user into memory.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import ijson


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "xuetangx_full"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "xuetangx_full"
BUFFER_SIZE = 8 * 1024 * 1024
COLUMNS = ("course_id", "user_id", "session_id", "action", "time")


@dataclass(frozen=True)
class SourceFile:
    input_name: str
    output_name: str
    expected_rows: int


@dataclass(frozen=True)
class ConversionRecord:
    source_file: str
    source_size_bytes: int
    output_file: str
    output_size_bytes: int
    row_count: int
    sha256: str


SOURCES = (
    SourceFile(
        "20150801-20151101-raw_user_activity.json",
        "activity_20150801_20151101.csv.gz",
        43_818_789,
    ),
    SourceFile(
        "20151101-20160201-raw_user_activity.json",
        "activity_20151101_20160201.csv.gz",
        59_430_567,
    ),
    SourceFile(
        "20160201-20160501-raw_user_activity.json",
        "activity_20160201_20160501.csv.gz",
        63_745_152,
    ),
    SourceFile(
        "20160501-20160801-raw_user_activity.json",
        "activity_20160501_20160801.csv.gz",
        61_676_966,
    ),
    SourceFile(
        "20160801-20170201-raw_user_activity.json",
        "activity_20160801_20170201.csv.gz",
        65_795_428,
    ),
    SourceFile(
        "20170201-20170801-raw_user_activity.json",
        "activity_20170201_20170801.csv.gz",
        56_985_474,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def rows_from_json(path: Path) -> Iterator[tuple[str, str, str, str, str]]:
    """Yield flat event rows while validating the published nested schema."""

    stack: list[str] = []
    course_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    event_values: list[str] | None = None

    with path.open("rb") as source:
        for _, event, value in ijson.parse(source):
            if event == "start_array":
                stack.append("array")
                if len(stack) == 6:
                    event_values = []
                continue

            if event == "start_map":
                stack.append("map")
                continue

            if event == "map_key":
                if len(stack) == 3:
                    user_id = str(value)
                elif len(stack) == 4:
                    session_id = str(value)
                continue

            if event == "string":
                if len(stack) == 2 and course_id is None:
                    course_id = value
                elif len(stack) == 6 and event_values is not None:
                    event_values.append(value)
                continue

            if event == "end_array":
                if len(stack) == 6:
                    if (
                        course_id is None
                        or user_id is None
                        or session_id is None
                        or event_values is None
                        or len(event_values) != 2
                    ):
                        raise ValueError(f"Malformed event in {path.name}")
                    yield course_id, user_id, session_id, event_values[0], event_values[1]
                    event_values = None
                elif len(stack) == 2:
                    course_id = None
                    user_id = None
                    session_id = None
                stack.pop()
                continue

            if event == "end_map":
                if len(stack) == 4:
                    session_id = None
                elif len(stack) == 3:
                    user_id = None
                stack.pop()

    if stack:
        raise ValueError(f"Incomplete JSON structure in {path.name}")


def convert(source: SourceFile, raw_dir: Path, output_dir: Path) -> ConversionRecord:
    input_path = raw_dir / source.input_name
    output_path = output_dir / source.output_name
    temporary = output_path.with_name(output_path.name + ".part")
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input JSON: {input_path}")
    temporary.unlink(missing_ok=True)

    rows = 0
    print(f"Convert: {source.input_name} -> {source.output_name}", flush=True)
    with temporary.open("wb") as raw_output:
        with gzip.GzipFile(fileobj=raw_output, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_output:
                writer = csv.writer(text_output, lineterminator="\n")
                writer.writerow(COLUMNS)
                for row in rows_from_json(input_path):
                    writer.writerow(row)
                    rows += 1
                    if rows % 5_000_000 == 0:
                        print(f"  {rows:,} rows", flush=True)

    if rows != source.expected_rows:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Unexpected row count for {source.input_name}: "
            f"expected {source.expected_rows:,}, got {rows:,}"
        )
    temporary.replace(output_path)
    return ConversionRecord(
        source_file=source.input_name,
        source_size_bytes=input_path.stat().st_size,
        output_file=source.output_name,
        output_size_bytes=output_path.stat().st_size,
        row_count=rows,
        sha256=sha256(output_path),
    )


def load_previous_manifest(output_dir: Path) -> dict:
    path = output_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def reusable_record(
    source: SourceFile,
    raw_dir: Path,
    output_dir: Path,
    previous: dict,
) -> ConversionRecord | None:
    input_path = raw_dir / source.input_name
    output_path = output_dir / source.output_name
    records = {
        item.get("output_file"): item
        for item in previous.get("files", [])
        if isinstance(item, dict)
    }
    item = records.get(source.output_name)
    if not input_path.is_file() or not output_path.is_file() or item is None:
        return None
    if (
        item.get("source_file") != source.input_name
        or item.get("source_size_bytes") != input_path.stat().st_size
        or item.get("output_size_bytes") != output_path.stat().st_size
        or item.get("row_count") != source.expected_rows
    ):
        return None
    return ConversionRecord(**item)


def write_manifest(output_dir: Path, records: list[ConversionRecord]) -> Path:
    manifest = {
        "dataset": "XuetangX full user activity",
        "format": "gzip-compressed CSV",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "columns": list(COLUMNS),
        "row_definition": "one row per [action, time] event in the source JSON",
        "files": [asdict(record) for record in records],
        "total_rows": sum(record.row_count for record in records),
        "total_compressed_bytes": sum(record.output_size_bytes for record in records),
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert six XuetangX full-log JSON files to streaming CSV.GZ."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Convert existing outputs again.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = load_previous_manifest(output_dir)
    records: list[ConversionRecord] = []

    for source in SOURCES:
        record = None if args.force else reusable_record(source, raw_dir, output_dir, previous)
        if record is None:
            record = convert(source, raw_dir, output_dir)
        else:
            print(f"Reuse converted CSV: {source.output_name}")
        records.append(record)

    manifest = write_manifest(output_dir, records)
    print(f"Ready: {output_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
