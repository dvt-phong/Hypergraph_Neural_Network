# 1. Download the raw XuetangX files (prediction_data, user_info, course_info).
# - XuetangX dataset http://moocdata.cn/data/user-activity

import argparse
import urllib.request
from importlib import import_module
from pathlib import Path
config = import_module("0_config")

# Download XuetangX dataset
def download_xuetangx(raw_dir=config.RAW):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for file_name, download_url in config.DOWNLOAD_FILES.items():
        destination_path = raw_dir / file_name
        if destination_path.is_file():
            print(f"Already exists, skipping {file_name}", flush=True)
            continue

        print(f"Downloading {file_name}", flush=True)
        urllib.request.urlretrieve(download_url, destination_path)


# Download OULAD dataset
def download_oulad(raw_dir=config.RAW):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the raw XuetangX dataset.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW)
    arguments = parser.parse_args()
    download_xuetangx(arguments.raw_dir)
