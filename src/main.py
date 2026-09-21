# Entry point và pipeline end-to-end được viết theo thứ tự đọc từ trên xuống.
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, pstdev

from config import (
    DATASET_CONTRACT,
    DEFAULT_FEATURE_SET,
    EXPERIMENT_SEEDS,
    FEATURE_SETS,
)
from paths import PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, REPORTS_DIR


# Mục đích: Chạy toàn bộ pipeline từ raw data đến test report cho một cấu hình.
# Đầu vào: Seed, feature set, graph/model settings, device và cache/test options.
# Đầu ra: Dictionary kết quả của từng phase cùng training/test reports.
# Lưu ý: Test chỉ chạy sau khi checkpoint đã được validation lựa chọn.
def run_pipeline(
    *,
    seed: int = 1,
    feature_set: str = DEFAULT_FEATURE_SET,
    behavioral_k: int = 10,
    epochs: int = 1,
    device: str = "auto",
    test_limit: int = 0,
    test_batch_size: int = 4,
    force: bool = False,
) -> dict[str, object]:
    from data.preprocess import prepare_dataset
    from data.split import build_splits
    from features.transform import build_features
    from hypergraph.construction import (
        build_hyperedges,
        build_initial_hypergraph,
    )
    from train import (
        HGSLTrainingConfig,
        evaluate_hgsl_checkpoint,
        run_hgsl_training,
    )

    prepared_data = prepare_dataset()
    data_splits = build_splits()
    node_features = build_features(force=force)
    hyperedges = build_hyperedges(force=force)
    initial_hypergraph = build_initial_hypergraph(
        behavioral_k=behavioral_k,
        force=force,
    )

    training_config = HGSLTrainingConfig(
        seed=seed,
        feature_set=feature_set,
        epochs=epochs,
        device=device,
    )
    training_report = run_hgsl_training(training_config)
    test_report = evaluate_hgsl_checkpoint(
        seed=seed,
        feature_set=feature_set,
        device_name=device,
        test_limit=test_limit,
        batch_size=test_batch_size,
        checkpoint_path=Path(training_report["checkpoint"]),
    )

    return {
        "prepared_data": prepared_data,
        "data_splits": data_splits,
        "node_features": node_features,
        "hyperedges": hyperedges,
        "initial_hypergraph": initial_hypergraph,
        "training": training_report,
        "test": test_report,
    }


# Mục đích: Train/test HGSL cho nhiều seed và feature ablation configs.
# Đầu vào: Tuple seeds/feature_sets, epochs, device và test protocol.
# Đầu ra: Report từng run và mean/std test metrics theo feature set.
# Lưu ý: Hàm dùng full validation và checkpoint path vừa tạo của chính run.
def run_experiments(
    *,
    seeds: tuple[int, ...] = EXPERIMENT_SEEDS,
    feature_sets: tuple[str, ...] = (DEFAULT_FEATURE_SET,),
    epochs: int = 1,
    device: str = "auto",
    test_limit: int = 0,
    test_batch_size: int = 4,
) -> dict[str, object]:
    from artifacts import utc_now, write_json_atomic
    from train import (
        HGSLTrainingConfig,
        evaluate_hgsl_checkpoint,
        run_hgsl_training,
    )

    runs: list[dict[str, object]] = []
    for feature_set in feature_sets:
        for seed in seeds:
            config = HGSLTrainingConfig(
                seed=seed,
                feature_set=feature_set,
                epochs=epochs,
                validation_limit=0,
                device=device,
            )
            training_report = run_hgsl_training(config)
            test_report = evaluate_hgsl_checkpoint(
                seed=seed,
                feature_set=feature_set,
                device_name=device,
                test_limit=test_limit,
                batch_size=test_batch_size,
                checkpoint_path=Path(training_report["checkpoint"]),
            )
            runs.append(
                {
                    "seed": seed,
                    "feature_set": feature_set,
                    "best_validation_metrics": training_report[
                        "best_validation_metrics"
                    ],
                    "test_metrics": test_report["test_metrics"],
                    "checkpoint": training_report["checkpoint"],
                }
            )

    metric_names = ("auc", "auprc", "f1", "precision", "recall")
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for feature_set in feature_sets:
        feature_runs = [
            run for run in runs if run["feature_set"] == feature_set
        ]
        summary[feature_set] = {}
        for metric_name in metric_names:
            values = [
                float(run["test_metrics"][metric_name])
                for run in feature_runs
            ]
            summary[feature_set][metric_name] = {
                "mean": mean(values),
                "std": pstdev(values),
            }

    report = {
        "generated_utc": utc_now(),
        "seeds": list(seeds),
        "feature_sets": list(feature_sets),
        "runs": runs,
        "test_summary": summary,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(REPORTS_DIR / "experiment_summary.json", report)
    return report


# Mục đích: In dataset contract dạng JSON cho CLI command contract.
# Đầu vào: Không có.
# Đầu ra: Exit code 0.
def _contract() -> int:
    print(json.dumps(DATASET_CONTRACT.to_dict(), ensure_ascii=False, indent=2))
    return 0


# Mục đích: In các thư mục chính mà pipeline đang sử dụng.
# Đầu vào: Không có.
# Đầu ra: Exit code 0.
def _structure() -> int:
    structure = {
        "project_root": str(PROJECT_ROOT),
        "raw_data": str(RAW_DATA_DIR),
        "processed_data": str(PROCESSED_DATA_DIR),
        "source_root": str(PROJECT_ROOT / "src"),
    }
    print(json.dumps(structure, ensure_ascii=False, indent=2))
    return 0


# Mục đích: Chạy Phase 1 preprocessing từ CLI và in report JSON.
# Đầu vào: Không có; dùng default project paths.
# Đầu ra: Exit code 0.
def _prepare_data() -> int:
    from data.preprocess import prepare_dataset

    result = prepare_dataset()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# Mục đích: Chạy Phase 2 user-disjoint split và in summary rút gọn.
# Đầu vào: Không có; dùng processed data mặc định.
# Đầu ra: Exit code 0.
def _split_data() -> int:
    from data.split import build_splits

    result = build_splits()
    summary = {
        seed: {
            split: {
                "users": values["users"],
                "enrollments": values["enrollments"],
                "dropout_rate": values["dropout_rate"],
            }
            for split, values in split_summary.items()
        }
        for seed, split_summary in result["summary_by_seed"].items()
    }
    output = {
        "artifact": str(PROCESSED_DATA_DIR / "splits.parquet"),
        "rows": result["artifact"]["rows"],
        "seeds": result["seeds"],
        "summary_by_seed": summary,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


# Mục đích: Chạy Phase 3 feature build từ CLI.
# Đầu vào: Cờ force cho phép bỏ qua cache.
# Đầu ra: Exit code 0 sau khi in artifact/audit summary.
def _build_features(*, force: bool) -> int:
    from features.transform import build_features

    manifest = build_features(force=force)
    output = {
        "behavior_artifact": str(PROCESSED_DATA_DIR / "X_base.npy"),
        "context_artifact": str(PROCESSED_DATA_DIR / "X_context.npy"),
        "cache_hit": manifest["cache_hit"],
        "seeds": manifest["seeds"],
        "behavior_shape": manifest["array_shape"],
        "context_shape": manifest["context_array_shape"],
        "feature_sets": {
            name: values["dimension"]
            for name, values in manifest["feature_sets"].items()
        },
        "dtype": manifest["array_dtype"],
        "raw_feature_audit": manifest["raw_feature_audit"],
        "raw_context_audit": manifest["raw_context_audit"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


# Mục đích: Chạy Phase 4 hyperedge candidate construction từ CLI.
# Đầu vào: Cờ force rebuild.
# Đầu ra: Exit code 0 sau khi in candidate/audit summary.
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


# Mục đích: Chạy Phase 5 materialize H0 và local memberships.
# Đầu vào: Cờ force và behavioral_k.
# Đầu ra: Exit code 0 sau khi in artifact summary.
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


# Mục đích: Chuyển CLI arguments thành BaselineConfig và train HGNN.
# Đầu vào: Seed, feature/model/optimizer/validation arguments.
# Đầu ra: Exit code 0 sau khi in training report.
def _train_baseline(
    *,
    seed: int,
    feature_set: str,
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
    from train import BaselineConfig, run_baseline_smoke

    report = run_baseline_smoke(
        BaselineConfig(
            seed=seed,
            feature_set=feature_set,
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


# Mục đích: Chạy một HGSL optimizer step để kiểm tra Phase 7.
# Đầu vào: Seed, feature/refinement/loss và device arguments.
# Đầu ra: Exit code 0 sau khi in smoke report.
def _check_hgsl(
    *,
    seed: int,
    feature_set: str,
    sampled_hyperedges: int,
    positive_nodes: int,
    negative_nodes: int,
    mode: str,
    top_r: int,
    threshold: float,
    contrastive_weight: float,
    device: str,
) -> int:
    from train import HGSLTrainingConfig, run_hgsl_smoke

    report = run_hgsl_smoke(
        HGSLTrainingConfig(
            seed=seed,
            feature_set=feature_set,
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


# Mục đích: Chuyển CLI arguments thành HGSLTrainingConfig và train model.
# Đầu vào: Toàn bộ model, refinement, loss, optimizer và validation settings.
# Đầu ra: Exit code 0 sau khi in Phase 8 report.
def _train_hgsl(
    *,
    seed: int,
    feature_set: str,
    epochs: int,
    patience: int,
    validation_limit: int,
    validation_batch_size: int,
    hidden_dim: int,
    dropout: float,
    sampled_hyperedges: int,
    positive_nodes: int,
    negative_nodes: int,
    mode: str,
    top_r: int,
    threshold: float,
    contrastive_weight: float,
    contrastive_temperature: float,
    contrastive_nodes: int,
    learning_rate: float,
    weight_decay: float,
    max_gradient_norm: float,
    device: str,
) -> int:
    from train import HGSLTrainingConfig, run_hgsl_training

    report = run_hgsl_training(
        HGSLTrainingConfig(
            seed=seed,
            feature_set=feature_set,
            epochs=epochs,
            patience=patience,
            validation_limit=validation_limit,
            validation_batch_size=validation_batch_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            sampled_hyperedges=sampled_hyperedges,
            positive_nodes=positive_nodes,
            negative_nodes=negative_nodes,
            mode=mode,
            top_r=top_r,
            threshold=threshold,
            contrastive_weight=contrastive_weight,
            contrastive_temperature=contrastive_temperature,
            contrastive_nodes=contrastive_nodes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_gradient_norm=max_gradient_norm,
            device=device,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# Mục đích: Đánh giá HGSL checkpoint từ CLI.
# Đầu vào: Experiment identity, device, test protocol và checkpoint tùy chọn.
# Đầu ra: Exit code 0 sau khi in Phase 9 report.
def _evaluate_hgsl(
    *,
    seed: int,
    feature_set: str,
    device: str,
    test_limit: int,
    batch_size: int,
    checkpoint: str | None,
) -> int:
    from train import evaluate_hgsl_checkpoint

    report = evaluate_hgsl_checkpoint(
        seed=seed,
        feature_set=feature_set,
        device_name=device,
        test_limit=test_limit,
        batch_size=batch_size,
        checkpoint_path=Path(checkpoint) if checkpoint else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# Mục đích: Đánh giá HGNN baseline checkpoint từ CLI.
# Đầu vào: Experiment identity, device, test protocol và checkpoint tùy chọn.
# Đầu ra: Exit code 0 sau khi in test report.
def _evaluate_baseline(
    *,
    seed: int,
    feature_set: str,
    device: str,
    test_limit: int,
    batch_size: int,
    checkpoint: str | None,
) -> int:
    from train import evaluate_baseline_checkpoint

    report = evaluate_baseline_checkpoint(
        seed=seed,
        feature_set=feature_set,
        device_name=device,
        test_limit=test_limit,
        batch_size=batch_size,
        checkpoint_path=Path(checkpoint) if checkpoint else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# Mục đích: Chạy batch experiment command và in summary mean/std.
# Đầu vào: Seeds, feature sets, epochs, device và test protocol.
# Đầu ra: Exit code 0.
def _run_experiments_command(
    *,
    seeds: tuple[int, ...],
    feature_sets: tuple[str, ...],
    epochs: int,
    device: str,
    test_limit: int,
    batch_size: int,
) -> int:
    report = run_experiments(
        seeds=seeds,
        feature_sets=feature_sets,
        epochs=epochs,
        device=device,
        test_limit=test_limit,
        test_batch_size=batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# Mục đích: Khai báo tất cả CLI commands và validate kiểu/range tham số.
# Đầu vào: Không có.
# Đầu ra: argparse.ArgumentParser hoàn chỉnh.
# Lưu ý: Hàm chỉ định nghĩa giao diện; không chạy phase nào.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XuetangX-247 hypergraph structure learning pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("contract", help="Print the locked dataset contract.")
    subparsers.add_parser("structure", help="Print the project paths used by the pipeline.")
    subparsers.add_parser(
        "prepare-data", help="Build the canonical Phase 1 Parquet artifacts."
    )
    subparsers.add_parser(
        "split-data", help="Build the locked user-disjoint experiment split."
    )
    features = subparsers.add_parser(
        "build-features",
        help="Build leakage-safe behavioral and context node features.",
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
    pipeline = subparsers.add_parser(
        "run-pipeline",
        help="Run preprocessing, features, hypergraph construction and HGSL training.",
    )
    pipeline.add_argument(
        "--seed", type=int, choices=(1, 11, 111, 1111, 11111), default=1
    )
    pipeline.add_argument(
        "--feature-set",
        type=lambda value: value.replace("-", "_"),
        choices=FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
    )
    pipeline.add_argument(
        "--behavioral-k", type=int, choices=(5, 10, 20), default=10
    )
    pipeline.add_argument("--epochs", type=int, default=1)
    pipeline.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    pipeline.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Test targets; zero evaluates the complete test split.",
    )
    pipeline.add_argument("--test-batch-size", type=int, default=4)
    pipeline.add_argument(
        "--force", action="store_true", help="Rebuild cached feature and graph stages."
    )
    baseline = subparsers.add_parser(
        "train-baseline",
        help="Run the Phase 6 full-batch HGNN smoke training.",
    )
    baseline.add_argument("--seed", type=int, choices=(1, 11, 111, 1111, 11111), default=1)
    baseline.add_argument(
        "--feature-set",
        type=lambda value: value.replace("-", "_"),
        choices=FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
    )
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
        default=0,
        help="Validation targets; zero uses the complete validation split.",
    )
    baseline.add_argument("--validation-batch-size", type=int, default=16)
    baseline.add_argument("--patience", type=int, default=5)
    hgsl = subparsers.add_parser(
        "check-hgsl",
        help="Run one end-to-end Phase 7 sparse-refinement optimizer step.",
    )
    hgsl.add_argument("--seed", type=int, choices=(1, 11, 111, 1111, 11111), default=1)
    hgsl.add_argument(
        "--feature-set",
        type=lambda value: value.replace("-", "_"),
        choices=FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
    )
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
    training = subparsers.add_parser(
        "train-hgsl",
        help="Train Phase 8 HGSL with validation-based checkpoint selection.",
    )
    training.add_argument(
        "--seed", type=int, choices=(1, 11, 111, 1111, 11111), default=1
    )
    training.add_argument(
        "--feature-set",
        type=lambda value: value.replace("-", "_"),
        choices=FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
    )
    training.add_argument("--epochs", type=int, default=1)
    training.add_argument("--patience", type=int, default=5)
    training.add_argument(
        "--validation-limit",
        type=int,
        default=0,
        help="Validation targets; zero uses the complete validation split.",
    )
    training.add_argument("--validation-batch-size", type=int, default=4)
    training.add_argument("--hidden-dim", type=int, default=64)
    training.add_argument("--dropout", type=float, default=0.5)
    training.add_argument("--sampled-hyperedges", type=int, default=96)
    training.add_argument("--positive-nodes", type=int, default=16)
    training.add_argument("--negative-nodes", type=int, default=16)
    training.add_argument("--mode", choices=("top_r", "threshold"), default="top_r")
    training.add_argument("--top-r", type=int, default=8)
    training.add_argument("--threshold", type=float, default=0.5)
    training.add_argument("--contrastive-weight", type=float, default=0.1)
    training.add_argument("--contrastive-temperature", type=float, default=0.2)
    training.add_argument("--contrastive-nodes", type=int, default=512)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--weight-decay", type=float, default=5e-4)
    training.add_argument("--max-gradient-norm", type=float, default=5.0)
    training.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    evaluation = subparsers.add_parser(
        "evaluate",
        help="Evaluate the validation-selected HGSL checkpoint on test data.",
    )
    evaluation.add_argument(
        "--seed", type=int, choices=(1, 11, 111, 1111, 11111), default=1
    )
    evaluation.add_argument(
        "--feature-set",
        type=lambda value: value.replace("-", "_"),
        choices=FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
    )
    evaluation.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    evaluation.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Test targets; zero evaluates the complete test split.",
    )
    evaluation.add_argument("--batch-size", type=int, default=4)
    evaluation.add_argument(
        "--checkpoint",
        help="Optional checkpoint path; otherwise use the latest training report.",
    )
    baseline_evaluation = subparsers.add_parser(
        "evaluate-baseline",
        help="Evaluate the validation-selected HGNN baseline on test data.",
    )
    baseline_evaluation.add_argument(
        "--seed", type=int, choices=EXPERIMENT_SEEDS, default=1
    )
    baseline_evaluation.add_argument(
        "--feature-set",
        type=lambda value: value.replace("-", "_"),
        choices=FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
    )
    baseline_evaluation.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    baseline_evaluation.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Test targets; zero evaluates the complete test split.",
    )
    baseline_evaluation.add_argument("--batch-size", type=int, default=16)
    baseline_evaluation.add_argument(
        "--checkpoint",
        help="Optional checkpoint path; otherwise use the latest training report.",
    )
    experiments = subparsers.add_parser(
        "run-experiments",
        help="Train and test multiple seeds and feature ablations.",
    )
    experiments.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=EXPERIMENT_SEEDS,
        default=list(EXPERIMENT_SEEDS),
    )
    experiments.add_argument(
        "--feature-sets",
        nargs="+",
        type=lambda value: value.replace("-", "_"),
        choices=FEATURE_SETS,
        default=[DEFAULT_FEATURE_SET],
    )
    experiments.add_argument("--epochs", type=int, default=1)
    experiments.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    experiments.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Test targets per run; zero evaluates complete test splits.",
    )
    experiments.add_argument("--batch-size", type=int, default=4)
    return parser


# Mục đích: Parse argv và dispatch đúng một CLI command.
# Đầu vào: Sequence arguments tùy chọn; None dùng sys.argv.
# Đầu ra: Process exit code của command.
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "contract":
        return _contract()
    if args.command == "structure":
        return _structure()
    if args.command == "prepare-data":
        return _prepare_data()
    if args.command == "split-data":
        return _split_data()
    if args.command == "build-features":
        return _build_features(force=args.force)
    if args.command == "build-hyperedges":
        return _build_hyperedges(force=args.force)
    if args.command == "build-hypergraph":
        return _build_hypergraph(
            force=args.force, behavioral_k=args.behavioral_k
        )
    if args.command == "run-pipeline":
        report = run_pipeline(
            seed=args.seed,
            feature_set=args.feature_set,
            behavioral_k=args.behavioral_k,
            epochs=args.epochs,
            device=args.device,
            test_limit=args.test_limit,
            test_batch_size=args.test_batch_size,
            force=args.force,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train-baseline":
        return _train_baseline(
            seed=args.seed,
            feature_set=args.feature_set,
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
            feature_set=args.feature_set,
            sampled_hyperedges=args.sampled_hyperedges,
            positive_nodes=args.positive_nodes,
            negative_nodes=args.negative_nodes,
            mode=args.mode,
            top_r=args.top_r,
            threshold=args.threshold,
            contrastive_weight=args.contrastive_weight,
            device=args.device,
        )
    if args.command == "train-hgsl":
        return _train_hgsl(
            seed=args.seed,
            feature_set=args.feature_set,
            epochs=args.epochs,
            patience=args.patience,
            validation_limit=args.validation_limit,
            validation_batch_size=args.validation_batch_size,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            sampled_hyperedges=args.sampled_hyperedges,
            positive_nodes=args.positive_nodes,
            negative_nodes=args.negative_nodes,
            mode=args.mode,
            top_r=args.top_r,
            threshold=args.threshold,
            contrastive_weight=args.contrastive_weight,
            contrastive_temperature=args.contrastive_temperature,
            contrastive_nodes=args.contrastive_nodes,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_gradient_norm=args.max_gradient_norm,
            device=args.device,
        )
    if args.command == "evaluate":
        return _evaluate_hgsl(
            seed=args.seed,
            feature_set=args.feature_set,
            device=args.device,
            test_limit=args.test_limit,
            batch_size=args.batch_size,
            checkpoint=args.checkpoint,
        )
    if args.command == "evaluate-baseline":
        return _evaluate_baseline(
            seed=args.seed,
            feature_set=args.feature_set,
            device=args.device,
            test_limit=args.test_limit,
            batch_size=args.batch_size,
            checkpoint=args.checkpoint,
        )
    if args.command == "run-experiments":
        return _run_experiments_command(
            seeds=tuple(args.seeds),
            feature_sets=tuple(args.feature_sets),
            epochs=args.epochs,
            device=args.device,
            test_limit=args.test_limit,
            batch_size=args.batch_size,
        )
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
