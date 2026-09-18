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


def _build_hypergraph(*, force: bool, behavioral_k: int) -> int:
    from hypergraph.construction import build_initial_hypergraph

    manifest = build_initial_hypergraph(
        force=force, behavioral_k=behavioral_k
    )
    output = {
        "artifacts": [
            str(PROCESSED_DATA_DIR / f"H0_train_seed_{seed}.npz")
            for seed in manifest["seeds"]
        ],
        "metadata": str(PROCESSED_DATA_DIR / "hyperedges.parquet"),
        "validation_memberships": str(
            PROCESSED_DATA_DIR / "validation_memberships.parquet"
        ),
        "test_memberships": str(
            PROCESSED_DATA_DIR / "test_memberships.parquet"
        ),
        "cache_hit": manifest["cache_hit"],
        "behavioral_k": manifest["behavioral_k"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _train_baseline(
    *,
    seed: int,
    epochs: int,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    device: str,
    validation_limit: int,
    validation_batch_size: int,
    patience: int,
) -> int:
    from training.trainer import BaselineConfig, run_baseline_smoke

    report = run_baseline_smoke(
        BaselineConfig(
            seed=seed,
            epochs=epochs,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            validation_limit=validation_limit,
            validation_batch_size=validation_batch_size,
            patience=patience,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _check_hgsl(
    *,
    seed: int,
    sampled_hyperedges: int,
    positive_nodes: int,
    negative_nodes: int,
    mode: str,
    top_r: int,
    threshold: float,
    contrastive_weight: float,
    device: str,
) -> int:
    from training.hgsl import HGSLSmokeConfig, run_hgsl_smoke

    report = run_hgsl_smoke(
        HGSLSmokeConfig(
            seed=seed,
            sampled_hyperedges=sampled_hyperedges,
            positive_nodes=positive_nodes,
            negative_nodes=negative_nodes,
            mode=mode,
            top_r=top_r,
            threshold=threshold,
            contrastive_weight=contrastive_weight,
            device=device,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
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
    graph = subparsers.add_parser(
        "build-hypergraph",
        help="Materialize sparse train H0 and inductive evaluation memberships.",
    )
    graph.add_argument(
        "--behavioral-k",
        type=int,
        choices=(5, 10, 20),
        default=10,
        help="Behavioral neighbors per anchor (default: 10).",
    )
    graph.add_argument(
        "--force", action="store_true", help="Rebuild H0 even when cached."
    )
    baseline = subparsers.add_parser(
        "train-baseline",
        help="Run the Phase 6 full-batch HGNN smoke training.",
    )
    baseline.add_argument("--seed", type=int, choices=(1, 11, 111, 1111, 11111), default=1)
    baseline.add_argument("--epochs", type=int, default=1)
    baseline.add_argument("--hidden-dim", type=int, default=64)
    baseline.add_argument("--dropout", type=float, default=0.5)
    baseline.add_argument("--learning-rate", type=float, default=0.01)
    baseline.add_argument("--weight-decay", type=float, default=5e-4)
    baseline.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    baseline.add_argument(
        "--validation-limit",
        type=int,
        default=128,
        help="Validation targets for a smoke run; zero uses all targets.",
    )
    baseline.add_argument("--validation-batch-size", type=int, default=16)
    baseline.add_argument("--patience", type=int, default=5)
    hgsl = subparsers.add_parser(
        "check-hgsl",
        help="Run one end-to-end Phase 7 sparse-refinement optimizer step.",
    )
    hgsl.add_argument("--seed", type=int, choices=(1, 11, 111, 1111, 11111), default=1)
    hgsl.add_argument("--sampled-hyperedges", type=int, default=96)
    hgsl.add_argument("--positive-nodes", type=int, default=16)
    hgsl.add_argument("--negative-nodes", type=int, default=16)
    hgsl.add_argument("--mode", choices=("top_r", "threshold"), default="top_r")
    hgsl.add_argument("--top-r", type=int, default=8)
    hgsl.add_argument("--threshold", type=float, default=0.5)
    hgsl.add_argument("--contrastive-weight", type=float, default=0.1)
    hgsl.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
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
    if args.command == "build-hypergraph":
        return _build_hypergraph(
            force=args.force, behavioral_k=args.behavioral_k
        )
    if args.command == "train-baseline":
        return _train_baseline(
            seed=args.seed,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            validation_limit=args.validation_limit,
            validation_batch_size=args.validation_batch_size,
            patience=args.patience,
        )
    if args.command == "check-hgsl":
        return _check_hgsl(
            seed=args.seed,
            sampled_hyperedges=args.sampled_hyperedges,
            positive_nodes=args.positive_nodes,
            negative_nodes=args.negative_nodes,
            mode=args.mode,
            top_r=args.top_r,
            threshold=args.threshold,
            contrastive_weight=args.contrastive_weight,
            device=args.device,
        )
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
