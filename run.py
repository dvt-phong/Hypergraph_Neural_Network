"""Repository-local entry point for the proposed hypergraph pipeline."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mooc_hgsl.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
