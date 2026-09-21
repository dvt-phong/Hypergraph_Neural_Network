# Entry point duy nhất để gọi CLI pipeline từ project root.
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from main import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
