"""Stable project paths used by the XuetangX-247 pipeline."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "xuetangx"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "xuetangx_247"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
RUNS_DIR = OUTPUTS_DIR / "runs"


def require_project_path(path: Path) -> Path:
    """Resolve a path and reject targets outside the project root."""

    resolved = path.resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"Path is outside the project root: {resolved}")
    return resolved
