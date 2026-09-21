# Tải và giải nén public XuetangX full activity logs theo cách restart-safe.
# Archive/JSON hợp lệ được tái sử dụng; file mới chỉ publish sau khi hash khớp.
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "xuetangx_full"
BASE_URL = "https://lfs.aminer.cn/misc/moocdata/data"
BUFFER_SIZE = 8 * 1024 * 1024


# Mục đích: Lưu tên, kích thước và hash kỳ vọng của một JSON member.
# Đầu vào: Metadata lấy từ public release.
# Đầu ra: Object bất biến dùng để xác minh sau giải nén.
@dataclass(frozen=True)
class JsonFile:
    name: str
    size_bytes: int
    sha256: str


# Mục đích: Mô tả một archive và các JSON members phải chứa.
# Đầu vào: Tên, size, hash và tuple JsonFile.
# Đầu ra: Object bất biến có property url.
@dataclass(frozen=True)
class Archive:
    name: str
    size_bytes: int
    sha256: str
    members: tuple[JsonFile, ...]

    @property
    # Mục đích: Tạo download URL đầy đủ cho archive.
    # Đầu vào: Archive name hiện tại.
    # Đầu ra: Chuỗi URL trên public data server.
    def url(self) -> str:
        return f"{BASE_URL}/{self.name}"


ARCHIVES = (
    Archive(
        name="20150801-20160801-activity.tar.gz",
        size_bytes=1_099_017_128,
        sha256="73a35e7062fa1279e803aeaea584e9c2c7fd7954820db63d6dd5f2d1901cf24a",
        members=(
            JsonFile(
                "20150801-20151101-raw_user_activity.json",
                3_923_261_699,
                "53217c810555914ae5bbf5d8dd15c5a8e52c5fa055f8126f1ea1577f0f0299fe",
            ),
            JsonFile(
                "20151101-20160201-raw_user_activity.json",
                5_337_752_939,
                "49f357c16aa4d048fc0393582a735dd7a15a6ab926c113daf5104223cfbd8533",
            ),
            JsonFile(
                "20160201-20160501-raw_user_activity.json",
                5_727_532_592,
                "8ab3e8a6bf8bfe9c0af324454ad7d2863d8c296a9423ece278b08eaaa030ffc0",
            ),
            JsonFile(
                "20160501-20160801-raw_user_activity.json",
                5_554_391_527,
                "cb3d95251cc02654777bc7c22df067042621a4f2ad4a1a665838df1d0d1ad522",
            ),
        ),
    ),
    Archive(
        name="20160801-20170801-activity.tar.gz",
        size_bytes=611_730_930,
        sha256="01994de20e62c424bff68f313e768fb01ed290367791efec6a1a56f5b4652c01",
        members=(
            JsonFile(
                "20160801-20170201-raw_user_activity.json",
                5_940_477_856,
                "71484265a86cf05270c5284c442f2bbb60d64aef425b4244b27615694ef09005",
            ),
            JsonFile(
                "20170201-20170801-raw_user_activity.json",
                5_141_900_717,
                "ec7335dba016f3ac6e1dd94402b53b6f6f67480ea18bdbf011d7c6184c249c4c",
            ),
        ),
    ),
)


# Mục đích: Tính SHA-256 của file lớn theo từng chunk.
# Đầu vào: File path.
# Đầu ra: Chuỗi SHA-256 hexadecimal.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


# Mục đích: Xác minh một file bằng tồn tại, kích thước và SHA-256.
# Đầu vào: Path, expected size và expected hash.
# Đầu ra: Boolean cho biết file khớp hoàn toàn hay không.
def matches(path: Path, *, size_bytes: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == size_bytes
        and sha256(path) == expected_sha256
    )


# Mục đích: Tải một archive hoặc dùng lại archive đã xác minh.
# Đầu vào: Archive spec, raw directory và force flag.
# Đầu ra: Path archive hoàn chỉnh.
# Lưu ý: Download vào .part và chỉ đổi tên sau khi size/hash khớp.
def download(archive: Archive, raw_dir: Path, *, force: bool) -> Path:
    destination = raw_dir / archive.name
    if not force and matches(
        destination,
        size_bytes=archive.size_bytes,
        expected_sha256=archive.sha256,
    ):
        print(f"Reuse verified archive: {archive.name}")
        return destination

    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        temporary.unlink()

    print(f"Download: {archive.url}", flush=True)
    request = urllib.request.Request(
        archive.url,
        headers={"User-Agent": "mooc-hgsl/0.1"},
    )
    downloaded = 0
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(BUFFER_SIZE):
            output.write(chunk)
            downloaded += len(chunk)
            print(
                f"  {downloaded / archive.size_bytes:6.1%} "
                f"({downloaded / 1024**2:,.0f} MiB)",
                end="\r",
                flush=True,
            )
    print()

    if not matches(
        temporary,
        size_bytes=archive.size_bytes,
        expected_sha256=archive.sha256,
    ):
        raise RuntimeError(f"Downloaded archive failed verification: {archive.name}")
    temporary.replace(destination)
    return destination


# Mục đích: Kiểm tra ổ đĩa đủ cho các JSON mới và file thay thế tạm.
# Đầu vào: Raw directory, archives và force flag.
# Đầu ra: Không có nếu đủ dung lượng; ném OSError nếu thiếu.
def ensure_disk_space(raw_dir: Path, archives: tuple[Archive, ...], *, force: bool) -> None:
    new_files = 0
    replacement_sizes: list[int] = []
    for archive in archives:
        for member in archive.members:
            target = raw_dir / member.name
            if not target.exists() or target.stat().st_size != member.size_bytes:
                new_files += member.size_bytes
            elif force:
                # Extraction is sequential, so only one replacement .part file
                # needs to coexist with the already extracted JSON files.
                replacement_sizes.append(member.size_bytes)

    free = shutil.disk_usage(raw_dir).free
    reserve = 2 * 1024**3
    required = new_files + max(replacement_sizes, default=0)
    if free < required + reserve:
        raise OSError(
            "Not enough free space to extract the JSON files: "
            f"need about {(required + reserve) / 1024**3:.1f} GiB, "
            f"have {free / 1024**3:.1f} GiB"
        )


# Mục đích: Stream-extract đúng các JSON đã khai báo khỏi tar.gz.
# Đầu vào: Archive spec/path, raw directory và force flag.
# Đầu ra: Không trả dữ liệu; tạo verified JSON files.
# Lưu ý: Chặn path traversal, duplicate member và hash mismatch.
def extract(archive: Archive, archive_path: Path, raw_dir: Path, *, force: bool) -> None:
    expected = {member.name: member for member in archive.members}
    found: set[str] = set()

    with tarfile.open(archive_path, mode="r:gz") as bundle:
        for tar_member in bundle:
            name = Path(tar_member.name).name
            if name not in expected:
                continue
            if name in found or not tar_member.isfile():
                raise ValueError(f"Invalid or duplicate archive member: {tar_member.name}")
            found.add(name)

            spec = expected[name]
            target = (raw_dir / name).resolve()
            if target.parent != raw_dir.resolve():
                raise ValueError(f"Unsafe archive member: {tar_member.name}")
            if not force and matches(
                target,
                size_bytes=spec.size_bytes,
                expected_sha256=spec.sha256,
            ):
                print(f"Reuse verified JSON: {name}")
                continue

            source = bundle.extractfile(tar_member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {tar_member.name}")

            temporary = target.with_name(target.name + ".part")
            digest = hashlib.sha256()
            written = 0
            print(f"Extract: {name}", flush=True)
            with source, temporary.open("wb") as output:
                while chunk := source.read(BUFFER_SIZE):
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)

            if written != spec.size_bytes or digest.hexdigest() != spec.sha256:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"Extracted JSON failed verification: {name}")
            temporary.replace(target)

    if found != set(expected):
        raise FileNotFoundError(
            f"{archive.name} is missing: {', '.join(sorted(set(expected) - found))}"
        )


# Mục đích: Ghi metadata nguồn và hashes của full activity release.
# Đầu vào: Raw directory chứa các archive/JSON đã xác minh.
# Đầu ra: Path manifest.json.
def write_manifest(raw_dir: Path) -> Path:
    manifest = {
        "dataset": "XuetangX full user activity",
        "source": "MoocData public release",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "archives": [],
    }
    for archive in ARCHIVES:
        manifest["archives"].append(
            {
                **{key: value for key, value in asdict(archive).items() if key != "members"},
                "url": archive.url,
                "members": [asdict(member) for member in archive.members],
            }
        )
    path = raw_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# Mục đích: Khai báo và parse CLI arguments của downloader.
# Đầu vào: sys.argv.
# Đầu ra: argparse.Namespace gồm raw-dir và force.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the two public XuetangX full-log archives."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--force", action="store_true", help="Download and extract again.")
    return parser.parse_args()


# Mục đích: Kiểm tra dung lượng, tải, giải nén và ghi manifest.
# Đầu vào: CLI arguments từ parse_args().
# Đầu ra: Không trả dữ liệu; in đường dẫn raw data/manifest.
def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    ensure_disk_space(raw_dir, ARCHIVES, force=args.force)

    for archive in ARCHIVES:
        archive_path = download(archive, raw_dir, force=args.force)
        extract(archive, archive_path, raw_dir, force=args.force)

    manifest = write_manifest(raw_dir)
    print(f"Ready: {raw_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
