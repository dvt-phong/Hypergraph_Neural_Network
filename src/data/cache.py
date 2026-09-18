"""Small cache and manifest helpers for processed XuetangX artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


BUFFER_SIZE = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
