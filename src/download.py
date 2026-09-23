# Download raw dataset XuetangX

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


def download(raw_dir=RAW):
    # Download dataset from url
    # If dataset downloaded, code will skip this
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name, url in FILES.items():
        path = raw_dir / name
        if path.is_file():
            print(f"Already exists, skipping {name}", flush=True)
            continue

        print(f"Downloading {name}", flush=True)
        urllib.request.urlretrieve(url, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    arguments = parser.parse_args()
    download(arguments.raw_dir)
