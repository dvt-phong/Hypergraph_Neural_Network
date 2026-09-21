# Khai báo tập trung các đường dẫn ổn định của project XuetangX-247.
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "xuetangx"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "xuetangx_247"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
RUNS_DIR = OUTPUTS_DIR / "runs"


# Mục đích: Bảo đảm một đường dẫn thao tác nằm bên trong project.
# Đầu vào: Path tương đối hoặc tuyệt đối cần kiểm tra.
# Đầu ra: Path tuyệt đối sau khi resolve.
# Lưu ý: Ném ValueError nếu đường dẫn trỏ ra ngoài PROJECT_ROOT.
def require_project_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"Path is outside the project root: {resolved}")
    return resolved
