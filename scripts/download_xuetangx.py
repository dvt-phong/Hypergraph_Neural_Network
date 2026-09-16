"""Tải XuetangX gốc từ MoocData vào data/raw/xuetangx.

Chạy: .venv/Scripts/python.exe scripts/download_xuetangx.py
Chỉ dùng thư viện chuẩn Python 3.12+. Đường dẫn không phụ thuộc nơi chạy lệnh.
"""

import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/xuetangx"
BASE_URL = "https://lfs.aminer.cn/misc/moocdata/data/"
FILES = ("prediction_data.tar.gz", "user_info.csv", "course_info.csv")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(name, previous):
    """Chỉ tái sử dụng file có SHA-256 khớp lần tải thành công trước."""
    target = DATA / name
    if target.exists() and previous.get("sha256") == sha256(target):
        print(f"Already downloaded: {name}", flush=True)
        return previous
    temporary = target.with_name(name + ".part")
    url = BASE_URL + name
    print(f"Downloading: {url}", flush=True)
    with urlopen(url, timeout=120) as response, temporary.open("wb") as out:
        expected = int(response.headers["Content-Length"])
        downloaded = 0
        next_progress = 25 * 1024 * 1024
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_progress:
                print(f"  {downloaded / expected:.0%}", flush=True)
                next_progress += 25 * 1024 * 1024
    if downloaded != expected:
        raise RuntimeError(f"Incomplete download: {name}")
    temporary.replace(target)
    return {"url": url, "bytes": downloaded, "sha256": sha256(target),
            "downloaded_utc": datetime.now(timezone.utc).isoformat()}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in FILES:
        manifest[name] = download(name, manifest.get(name, {}))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Chỉ lấy 4 CSV cần thiết; không giải nén đường dẫn tùy ý từ archive.
    required = {f"{split}_{kind}.csv" for split in ("train", "test")
                for kind in ("log", "truth")}
    found = set()
    with tarfile.open(DATA / FILES[0], "r:gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if name not in required or not member.isfile():
                continue
            if name in found:
                raise ValueError(f"Duplicate archive member: {name}")
            found.add(name)
            print(f"Extracting: {name}", flush=True)
            temporary = DATA / (name + ".part")
            with archive.extractfile(member) as source, temporary.open("wb") as out:
                shutil.copyfileobj(source, out)
            temporary.replace(DATA / name)
    if found != required:
        raise ValueError(f"Missing files: {required - found}")
    print(f"Done: {DATA}", flush=True)


if __name__ == "__main__":
    main()
