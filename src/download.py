# Download the raw XuetangX dataset.

import argparse
import urllib.request
from pathlib import Path

# Get root path to contain files
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "xuetangx"
FILES = {
    "prediction_data.tar.gz":
        "https://lfs.aminer.cn/misc/moocdata/data/prediction_data.tar.gz",
    "user_info.csv":
        "https://lfs.aminer.cn/misc/moocdata/data/user_info.csv",
    "course_info.csv":
        "https://lfs.aminer.cn/misc/moocdata/data/course_info.csv",
}
CLI_DESCRIPTION = "Download the raw XuetangX dataset."


# Download every missing raw file into the requested directory.
def download(raw_dir=RAW):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for file_name, download_url in FILES.items():
        destination_path = raw_dir / file_name
        if destination_path.is_file():
            print(f"Already exists, skipping {file_name}", flush=True)
            continue

        print(f"Downloading {file_name}", flush=True)
        urllib.request.urlretrieve(download_url, destination_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    arguments = parser.parse_args()
    download(arguments.raw_dir)
