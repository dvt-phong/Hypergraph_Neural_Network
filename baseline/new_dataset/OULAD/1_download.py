# 1. Download the raw OULAD archive (7 CSV files in one zip).
# - OULAD https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset

import argparse
import urllib.request
import zipfile
from importlib import import_module
from pathlib import Path
config = import_module("0_config")


# Download OULAD dataset (skipped if the zip exists), then check that it has the 7 CSV files.
def download_oulad(raw_dir=config.RAW):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    destination_path = raw_dir / config.ZIP_FILE
    if destination_path.is_file():
        print(f"Already exists, skipping {config.ZIP_FILE}", flush=True)
    else:
        print(f"Downloading {config.ZIP_FILE}", flush=True)
        urllib.request.urlretrieve(config.DOWNLOAD_URL, destination_path)

    with zipfile.ZipFile(destination_path) as archive:
        names = set(archive.namelist())
    missing_files = []
    for file_name in config.RAW_FILES:
        if file_name not in names:
            missing_files.append(file_name)
    if missing_files:
        raise ValueError(f"{destination_path} lacks {missing_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the raw OULAD dataset.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW)
    arguments = parser.parse_args()
    download_oulad(arguments.raw_dir)
