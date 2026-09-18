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


def _audit_data() -> int:
    from data.audit import audit_dataset

    audit = audit_dataset()
    summary = {
        "artifact": str(PROCESSED_DATA_DIR / "audit.json"),
        "events": audit["logs"]["events"],
        "enrollments": audit["logs"]["enrollments"],
        "users": audit["logs"]["users"],
        "courses": audit["logs"]["courses"],
        "retained_events_35d": audit["temporal"]["retained_day_0_to_34"],
        "duplicate_rows_beyond_first": audit["duplicates"][
            "duplicate_rows_beyond_first"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _prepare_data(*, force: bool) -> int:
    from data.preprocess import prepare_dataset

    manifest = prepare_dataset(force=force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _split_data(*, force: bool) -> int:
    from data.split import build_splits

    manifest = build_splits(force=force)
    summary = {
        seed: {
            split: {
                "users": values["users"],
                "enrollments": values["enrollments"],
                "dropout_rate": values["dropout_rate"],
            }
            for split, values in split_summary.items()
        }
        for seed, split_summary in manifest["summary_by_seed"].items()
    }
    output = {
        "artifact": str(PROCESSED_DATA_DIR / "splits.parquet"),
        "cache_hit": manifest["cache_hit"],
        "seeds": manifest["seeds"],
        "sha256": manifest["artifact"]["sha256"],
        "summary_by_seed": summary,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _build_features(*, force: bool) -> int:
    from features.transform import build_features

    manifest = build_features(force=force)
    output = {
        "artifact": str(PROCESSED_DATA_DIR / "X_base.npy"),
        "cache_hit": manifest["cache_hit"],
        "seeds": manifest["seeds"],
        "shape": manifest["array_shape"],
        "dtype": manifest["array_dtype"],
        "raw_feature_audit": manifest["raw_feature_audit"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _build_hyperedges(*, force: bool) -> int:
    from hypergraph.construction import build_hyperedges

    manifest = build_hyperedges(force=force)
    output = {
        "structural_artifact": str(PROCESSED_DATA_DIR / "structural_memberships.parquet"),
        "behavioral_artifact": str(PROCESSED_DATA_DIR / "behavioral_neighbors.npz"),
        "audit": str(PROCESSED_DATA_DIR / "hyperedge_audit.json"),
        "cache_hit": manifest["cache_hit"],
        "families": manifest["families"],
        "k_candidates": manifest["k_candidates"],
        "behavioral_shape": manifest["behavioral_array_shape"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XuetangX-247 hypergraph structure learning pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("contract", help="Print the locked dataset contract.")
    subparsers.add_parser("structure", help="Print the project paths used by the pipeline.")
    subparsers.add_parser("audit-data", help="Audit all labeled XuetangX source files.")
    prepare = subparsers.add_parser(
        "prepare-data", help="Build the canonical Phase 1 Parquet artifacts."
    )
    prepare.add_argument(
        "--force", action="store_true", help="Rebuild artifacts even when the cache is valid."
    )
    split = subparsers.add_parser(
        "split-data", help="Build the locked user-disjoint experiment split."
    )
    split.add_argument(
        "--force", action="store_true", help="Rebuild the split even when the cache is valid."
    )
    features = subparsers.add_parser(
        "build-features", help="Build leakage-safe X_base for all experiment seeds."
    )
    features.add_argument(
        "--force", action="store_true", help="Rebuild features even when the cache is valid."
    )
    hyperedges = subparsers.add_parser(
        "build-hyperedges",
        help="Build Phase 4 Course, Object and Behavioral candidates.",
    )
    hyperedges.add_argument(
        "--force", action="store_true", help="Rebuild hyperedges even when cached."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "contract":
        return _contract()
    if args.command == "structure":
        return _structure()
    if args.command == "audit-data":
        return _audit_data()
    if args.command == "prepare-data":
        return _prepare_data(force=args.force)
    if args.command == "split-data":
        return _split_data(force=args.force)
    if args.command == "build-features":
        return _build_features(force=args.force)
    if args.command == "build-hyperedges":
        return _build_hyperedges(force=args.force)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
