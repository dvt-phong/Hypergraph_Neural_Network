"""Command-line entry point for the phased XuetangX-247 implementation."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from data.schema import DATASET_CONTRACT
from paths import PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR


def _contract() -> int:
    print(json.dumps(DATASET_CONTRACT.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _structure() -> int:
    structure = {
        "project_root": str(PROJECT_ROOT),
        "raw_data": str(RAW_DATA_DIR),
        "processed_data": str(PROCESSED_DATA_DIR),
        "source_root": str(PROJECT_ROOT / "src"),
    }
    print(json.dumps(structure, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XuetangX-247 hypergraph structure learning pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("contract", help="Print the locked dataset contract.")
    subparsers.add_parser("structure", help="Print the project paths used by the pipeline.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "contract":
        return _contract()
    if args.command == "structure":
        return _structure()
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
