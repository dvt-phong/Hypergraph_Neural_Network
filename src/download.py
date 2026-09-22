"""Download and unpack the XuetangX files used by this project."""

import argparse
import csv
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "xuetangx"
FILES = {
    "prediction_data.tar.gz": (
        "https://lfs.aminer.cn/misc/moocdata/data/prediction_data.tar.gz",
        "a8c32b49dcba55673a8d3badb353848290dbb17b0c8b897e3d9e3dcf8df584c2",
    ),
    "user_info.csv": (
        "https://lfs.aminer.cn/misc/moocdata/data/user_info.csv",
        "f9330170a7e1097391dd4fa4e995e780bfeed374e78f7ea8290bdc92c7f1a510",
    ),
    "course_info.csv": (
        "https://lfs.aminer.cn/misc/moocdata/data/course_info.csv",
        "eb2d6f4092202247b508701c51f8ed38c189f818619df15eca0b30396cee7f72",
    ),
}
ARCHIVE_FILES = (
    "train_log.csv", "test_log.csv", "train_truth.csv", "test_truth.csv"
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw_dir=RAW):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, expected_hash) in FILES.items():
        path = raw_dir / name
        if not path.exists() or sha256(path) != expected_hash:
            print(f"Downloading {name}", flush=True)
            temporary = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, temporary)
            if sha256(temporary) != expected_hash:
                temporary.unlink()
                raise ValueError(f"SHA-256 does not match for {name}")
            temporary.replace(path)

    with tarfile.open(raw_dir / "prediction_data.tar.gz", "r:gz") as archive:
        members = {Path(member.name).name: member for member in archive if member.isfile()}
        for name in ARCHIVE_FILES:
            destination = raw_dir / name
            if destination.exists():
                continue
            if name not in members:
                raise FileNotFoundError(f"{name} is missing from the archive")
            source = archive.extractfile(members[name])
            if source is None:
                raise OSError(f"Cannot extract {name}")
            with source, open(destination, "wb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
            print(f"Extracted {name}", flush=True)


def json_to_csv(source, destination):
    """Optional helper for a JSON array or JSONL file; XuetangX input is already CSV."""
    source = Path(source)
    with open(source, encoding="utf-8") as stream:
        if source.suffix == ".jsonl":
            records = (json.loads(line) for line in stream if line.strip())
        else:
            records = iter(json.load(stream))
        first = next(records, None)
        if first is None:
            raise ValueError("The JSON file is empty")
        with open(destination, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(first))
            writer.writeheader()
            writer.writerow(first)
            writer.writerows(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    arguments = parser.parse_args()
    download(arguments.raw_dir)
