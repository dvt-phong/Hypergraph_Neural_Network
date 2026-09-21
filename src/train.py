# Huấn luyện, validation, checkpoint và evaluation cho HGNN/HGSL.
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import duckdb
import numpy as np
import torch

from artifacts import source_signature, sql_path, utc_now, write_json_atomic
from config import (
    DEFAULT_FEATURE_SET,
    EXPERIMENT_SEEDS,
    FEATURE_SETS,
    feature_columns,
)
from features.transform import FEATURE_MANIFEST
from hypergraph.construction import (
    GRAPH_MANIFEST_ARTIFACT,
    HYPEREDGE_METADATA_ARTIFACT,
)
from graph_data import (
    LocalGraphStore,
    batch_local_hypergraphs,
    load_train_hypergraph,
)
from hsl import (
    HGSLModel,
    RefinementConfig,
    classification_loss,
    hgsl_objective,
    train_pos_weight,
)
from model import HGNNBaseline, HypergraphOperator
from paths import PROCESSED_DATA_DIR, REPORTS_DIR, RUNS_DIR, require_project_path
from metrics import binary_metrics


# Mục đích: Gom toàn bộ tham số của một run HGNN baseline.
# Đầu vào: Seed, feature/model/optimizer settings và validation protocol.
# Đầu ra: Config bất biến được train và checkpoint cùng sử dụng.
@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 1
    feature_set: str = DEFAULT_FEATURE_SET
    hidden_dim: int = 64
    dropout: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 1
    device: str = "auto"
    validation_limit: int = 0
    validation_batch_size: int = 16
    patience: int = 5

    # Mục đích: Kiểm tra baseline config trước khi cấp phát model/data lớn.
    # Đầu vào: Các field của config hiện tại.
    # Đầu ra: Không có nếu hợp lệ; ném ValueError nếu sai.
    def validate(self) -> None:
        if self.seed not in EXPERIMENT_SEEDS:
            raise ValueError(f"seed must be one of {EXPERIMENT_SEEDS}")
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        if self.hidden_dim <= 0 or self.epochs <= 0:
            raise ValueError("hidden_dim and epochs must be positive")
        if self.validation_limit < 0 or self.validation_limit == 1:
            raise ValueError("validation_limit must be zero or at least two")
        if self.validation_batch_size <= 0 or self.patience <= 0:
            raise ValueError("validation_batch_size and patience must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")


# Mục đích: Gom toàn bộ tham số để tái lập một HGSL experiment.
# Đầu vào: Seed, model/refinement/loss/optimizer và validation settings.
# Đầu ra: Config bất biến được lưu nguyên trong checkpoint.
@dataclass(frozen=True)
class HGSLTrainingConfig:
    seed: int = 1
    feature_set: str = DEFAULT_FEATURE_SET
    hidden_dim: int = 64
    dropout: float = 0.5
    sampled_hyperedges: int = 96
    positive_nodes: int = 16
    negative_nodes: int = 16
    mode: str = "top_r"
    top_r: int = 8
    threshold: float = 0.5
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.2
    contrastive_nodes: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    max_gradient_norm: float = 5.0
    epochs: int = 1
    patience: int = 5
    validation_limit: int = 0
    validation_batch_size: int = 4
    device: str = "auto"

    # Mục đích: Chọn riêng các field cần cho HypergraphRefiner.
    # Đầu vào: Config hiện tại.
    # Đầu ra: RefinementConfig tương ứng.
    def refinement_config(self) -> RefinementConfig:
        return RefinementConfig(
            sampled_hyperedges=self.sampled_hyperedges,
            positive_nodes=self.positive_nodes,
            negative_nodes=self.negative_nodes,
            mode=self.mode,
            top_r=self.top_r,
            threshold=self.threshold,
        )

    # Mục đích: Kiểm tra toàn bộ HGSL training protocol.
    # Đầu vào: Các field của config hiện tại.
    # Đầu ra: Không có nếu hợp lệ; ném ValueError nếu sai.
    def validate(self) -> None:
        if self.seed not in EXPERIMENT_SEEDS:
            raise ValueError(f"seed must be one of {EXPERIMENT_SEEDS}")
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        if self.hidden_dim <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid model dimensions or dropout")
        if self.contrastive_weight < 0 or self.contrastive_temperature <= 0:
            raise ValueError("Invalid contrastive parameters")
        if self.contrastive_nodes < 2:
            raise ValueError("contrastive_nodes must be at least two")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        if self.max_gradient_norm <= 0 or self.epochs <= 0 or self.patience <= 0:
            raise ValueError("Training counts and gradient norm must be positive")
        if self.validation_limit < 0 or self.validation_limit == 1:
            raise ValueError("validation_limit must be zero or at least two")
        if self.validation_batch_size <= 0:
            raise ValueError("validation_batch_size must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        self.refinement_config().validate()


# Mục đích: Chọn CPU/CUDA device thực sự dùng cho run.
# Đầu vào: "auto", "cpu" hoặc "cuda".
# Đầu ra: torch.device đã resolve.
# Lưu ý: Yêu cầu cuda khi máy không có CUDA sẽ báo lỗi rõ ràng.
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


# Mục đích: Cố định RNG của Python, NumPy và PyTorch cho một run.
# Đầu vào: Integer experiment seed.
# Đầu ra: Không trả dữ liệu; cập nhật global RNG states.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Mục đích: Tạo ID ngắn đại diện cho training config.
# Đầu vào: BaselineConfig hoặc HGSLTrainingConfig.
# Đầu ra: 10 ký tự đầu của SHA-256 config JSON.
# Lưu ý: Device bị bỏ khỏi ID vì không thay đổi định nghĩa experiment.
def configuration_id(config: BaselineConfig | HGSLTrainingConfig) -> str:
    values = asdict(config)
    values.pop("device")
    encoded = json.dumps(values, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


# Mục đích: Nhận diện cả config lẫn đúng graph/features dùng trong run.
# Đầu vào: Config và hai manifest paths.
# Đầu ra: Chuỗi configuration_id_graphhash_featurehash.
# Lưu ý: Đổi behavioral_k hoặc rebuild feature sẽ tạo experiment ID mới.
def experiment_id(
    config: BaselineConfig | HGSLTrainingConfig,
    graph_manifest_path: Path,
    feature_manifest_path: Path,
) -> str:
    graph_hash = source_signature(graph_manifest_path, include_hash=True)["sha256"]
    feature_hash = source_signature(
        feature_manifest_path, include_hash=True
    )["sha256"]
    return (
        f"{configuration_id(config)}_"
        f"{graph_hash[:8]}_{feature_hash[:8]}"
    )


# Mục đích: Ngăn checkpoint chạy trên artifact khác phiên bản đã train.
# Đầu vào: Checkpoint dictionary và processed output directory hiện tại.
# Đầu ra: Không có nếu hai SHA-256 đều khớp.
# Lưu ý: Ném RuntimeError khi thiếu hash hoặc graph/feature manifest đã đổi.
def validate_checkpoint_artifacts(
    checkpoint: dict[str, Any],
    output_dir: Path,
) -> None:
    artifact_paths = {
        "graph_manifest": output_dir / GRAPH_MANIFEST_ARTIFACT,
        "feature_manifest": output_dir / FEATURE_MANIFEST,
    }
    for checkpoint_key, current_path in artifact_paths.items():
        recorded = checkpoint.get(checkpoint_key)
        if not isinstance(recorded, dict) or "sha256" not in recorded:
            raise RuntimeError(
                f"Checkpoint does not contain a hash for {checkpoint_key}"
            )
        current_hash = source_signature(current_path, include_hash=True)["sha256"]
        if recorded["sha256"] != current_hash:
            raise RuntimeError(
                f"Checkpoint {checkpoint_key} does not match current artifacts"
            )


# Mục đích: Tìm checkpoint explicit hoặc checkpoint mới nhất ghi trong report.
# Đầu vào: Model name, seed, feature_set, report/runs dirs và path tùy chọn.
# Đầu ra: Path checkpoint tồn tại trong project.
# Lưu ý: Có fallback tên legacy để đọc checkpoint cũ khi chưa có report mới.
def _checkpoint_from_report(
    *,
    model_name: str,
    seed: int,
    feature_set: str,
    report_dir: Path,
    runs_dir: Path,
    checkpoint_path: Path | None,
) -> Path:
    if checkpoint_path is not None:
        resolved = require_project_path(checkpoint_path)
    else:
        phase = 6 if model_name == "hgnn" else 8
        report_name = f"phase{phase}_{model_name}_{feature_set}_seed_{seed}.json"
        latest_report = report_dir / report_name
        if latest_report.is_file():
            report = json.loads(latest_report.read_text(encoding="utf-8"))
            resolved = require_project_path(Path(report["checkpoint"]))
        else:
            legacy_name = (
                f"hgnn_baseline_{feature_set}_seed_{seed}.pt"
                if model_name == "hgnn"
                else f"hgsl_{feature_set}_seed_{seed}.pt"
            )
            resolved = runs_dir / legacy_name
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint {resolved}. Train the model first."
        )
    return resolved


# Mục đích: Đọc binary labels cho một danh sách global node IDs.
# Đầu vào: nodes.parquet và node_ids theo thứ tự cần trả.
# Đầu ra: Float32 label vector cùng thứ tự node_ids.
def labels_for_nodes(nodes_path: Path, node_ids: np.ndarray) -> np.ndarray:
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""SELECT node_id, label
                FROM read_parquet('{sql_path(nodes_path)}') ORDER BY node_id"""
        ).fetchnumpy()
    finally:
        connection.close()
    all_labels = np.empty(rows["node_id"].size, dtype=np.float32)
    all_labels[rows["node_id"].astype(np.int64, copy=False)] = rows["label"]
    labels = all_labels[node_ids]
    if not np.all((labels == 0) | (labels == 1)):
        raise RuntimeError("Loaded labels are not binary")
    return labels


# Mục đích: Chọn toàn bộ hoặc một subset validation/test cố định.
# Đầu vào: LocalGraphStore, nodes path, limit và seed.
# Đầu ra: Target node IDs và labels tương ứng.
# Lưu ý: Subset nhỏ vẫn được điều chỉnh để có đủ hai lớp tính AUC/AUPRC.
def select_evaluation_targets(
    store: LocalGraphStore,
    nodes_path: Path,
    *,
    limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    targets = store.target_node_ids()
    labels = labels_for_nodes(nodes_path, targets)
    if limit and limit < targets.size:
        generator = np.random.default_rng(seed)
        selected = np.sort(generator.choice(targets.size, size=limit, replace=False))
        if np.unique(labels[selected]).size < 2:
            missing_class = 1 - int(labels[selected[0]])
            replacement = np.flatnonzero(labels == missing_class)
            if replacement.size == 0:
                raise RuntimeError("Validation split contains only one label class")
            selected[-1] = replacement[0]
            selected.sort()
        targets = targets[selected]
        labels = labels[selected]
    return targets, labels


# Mục đích: Chạy HGNN baseline trên independent local graphs và tính metrics.
# Đầu vào: Model, graph store, target IDs/labels, batch size và device.
# Đầu ra: Dictionary AUC/AUPRC/F1/precision/recall.
# Lưu ý: Block-diagonal batching không tạo message giữa các target.
def _evaluate_local(
    model: HGNNBaseline,
    store: LocalGraphStore,
    target_node_ids: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, target_node_ids.size, batch_size):
            batch_ids = target_node_ids[start : start + batch_size]
            batch = batch_local_hypergraphs(store.load_many(batch_ids))
            features = torch.as_tensor(
                batch.features, dtype=torch.float32, device=device
            )
            operator = HypergraphOperator.from_scipy(
                batch.incidence, device=device
            )
            logits, _ = model(features, operator)
            target_indices = torch.as_tensor(
                batch.target_indices, dtype=torch.int64, device=device
            )
            probabilities.append(
                torch.sigmoid(logits[target_indices]).cpu().numpy()
            )
    return binary_metrics(labels, np.concatenate(probabilities))


# Mục đích: Lưu atomic HGNN checkpoint khi validation AUC tốt hơn.
# Đầu vào: Path, model, config, epoch, metrics và artifact manifests.
# Đầu ra: Không trả dữ liệu; tạo file .pt.
# Lưu ý: Checkpoint chứa hashes và feature schema để kiểm tra khi load.
def _write_checkpoint_atomic(
    path: Path,
    model: HGNNBaseline,
    config: BaselineConfig,
    epoch: int,
    validation_metrics: dict[str, float],
    graph_manifest_path: Path,
    feature_manifest_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    state = {
        "phase": 6,
        "epoch": epoch,
        "config": asdict(config),
        "validation_metrics": validation_metrics,
        "graph_manifest": source_signature(
            graph_manifest_path, include_hash=True
        ),
        "feature_manifest": source_signature(
            feature_manifest_path, include_hash=True
        ),
        "feature_names": list(feature_columns(config.feature_set)),
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }
    torch.save(state, temporary)
    temporary.replace(path)


# Mục đích: Train full-batch HGNN baseline và chọn epoch bằng validation AUC.
# Đầu vào: BaselineConfig cùng output/report/runs directories.
# Đầu ra: Report chứa history, best checkpoint, metrics và artifact hashes.
# Lưu ý: Tên giữ hậu tố smoke để tương thích code cũ; validation_limit=0 là full run.
def run_baseline_smoke(
    config: BaselineConfig,
    output_dir: Path = PROCESSED_DATA_DIR,
    report_dir: Path = REPORTS_DIR,
    runs_dir: Path = RUNS_DIR,
) -> dict[str, Any]:
    config.validate()
    output_dir = require_project_path(output_dir)
    report_dir = require_project_path(report_dir)
    runs_dir = require_project_path(runs_dir)
    set_seed(config.seed)
    device = resolve_device(config.device)
    graph = load_train_hypergraph(
        config.seed, output_dir, feature_set=config.feature_set
    )
    labels_numpy = labels_for_nodes(output_dir / "nodes.parquet", graph.node_ids)
    features = torch.as_tensor(graph.features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_numpy, dtype=torch.float32, device=device)
    operator = HypergraphOperator.from_scipy(graph.incidence, device=device)
    model = HGNNBaseline(
        input_dim=features.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    pos_weight = train_pos_weight(labels)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    losses: list[float] = []
    gradient_norms: list[float] = []
    validation_history: list[dict[str, Any]] = []
    best_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    graph_manifest_path = output_dir / GRAPH_MANIFEST_ARTIFACT
    feature_manifest_path = output_dir / FEATURE_MANIFEST
    run_id = experiment_id(config, graph_manifest_path, feature_manifest_path)
    checkpoint_path = runs_dir / (
        f"hgnn_baseline_{config.feature_set}_seed_{config.seed}_{run_id}.pt"
    )
    with LocalGraphStore(
        config.seed,
        "validation",
        output_dir,
        feature_set=config.feature_set,
    ) as validation_store:
        validation_node_ids, validation_labels = select_evaluation_targets(
            validation_store,
            output_dir / "nodes.parquet",
            limit=config.validation_limit,
            seed=config.seed,
        )
        for epoch in range(1, config.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features, operator)
            loss = classification_loss(logits, labels, pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("HGNN loss is non-finite")
            loss.backward()
            squared_norm = torch.zeros((), dtype=torch.float32, device=device)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("HGNN gradient is non-finite")
                    squared_norm += torch.sum(parameter.grad.square())
            gradient_norm = torch.sqrt(squared_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))

            validation_metrics = _evaluate_local(
                model,
                validation_store,
                validation_node_ids,
                validation_labels,
                batch_size=config.validation_batch_size,
                device=device,
            )
            validation_history.append(
                {"epoch": epoch, **validation_metrics}
            )
            if validation_metrics["auc"] > best_auc:
                best_auc = validation_metrics["auc"]
                best_epoch = epoch
                epochs_without_improvement = 0
                _write_checkpoint_atomic(
                    checkpoint_path,
                    model,
                    config,
                    epoch,
                    validation_metrics,
                    graph_manifest_path,
                    feature_manifest_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break

    model.eval()
    with torch.no_grad():
        logits, embeddings = model(features, operator)
        probabilities = torch.sigmoid(logits).cpu().numpy()
    metrics = binary_metrics(labels_numpy, probabilities)
    report = {
        "phase": 6,
        "scope": (
            "HGNN training with complete validation"
            if config.validation_limit == 0
            else "HGNN smoke training with sampled validation"
        ),
        "generated_utc": utc_now(),
        "configuration_id": configuration_id(config),
        "experiment_id": run_id,
        "torch_version": torch.__version__,
        "config": asdict(config),
        "resolved_device": str(device),
        "nodes": int(graph.incidence.shape[0]),
        "hyperedges": int(graph.incidence.shape[1]),
        "incidences": int(graph.incidence.nnz),
        "feature_dim": int(features.shape[1]),
        "feature_names": list(feature_columns(config.feature_set)),
        "embedding_dim": int(embeddings.shape[1]),
        "positive_labels": int(labels_numpy.sum()),
        "negative_labels": int(labels_numpy.size - labels_numpy.sum()),
        "pos_weight": float(pos_weight.detach().cpu()),
        "loss_by_epoch": losses,
        "gradient_norm_by_epoch": gradient_norms,
        "train_metrics": metrics,
        "validation_targets": int(validation_node_ids.size),
        "validation_sampling": (
            "all" if config.validation_limit == 0 else "fixed random subset"
        ),
        "validation_history": validation_history,
        "best_epoch": best_epoch,
        "best_validation_metrics": validation_history[best_epoch - 1],
        "checkpoint": str(checkpoint_path),
        "checkpoint_graph_manifest_sha256": source_signature(
            graph_manifest_path, include_hash=True
        )["sha256"],
        "checkpoint_feature_manifest_sha256": source_signature(
            feature_manifest_path, include_hash=True
        )["sha256"],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    latest_report_path = (
        report_dir / f"phase6_hgnn_{config.feature_set}_seed_{config.seed}.json"
    )
    detailed_report_path = report_dir / (
        f"phase6_hgnn_{config.feature_set}_seed_{config.seed}_{run_id}.json"
    )
    write_json_atomic(latest_report_path, report)
    write_json_atomic(detailed_report_path, report)
    return report


# Mục đích: Đánh giá checkpoint HGNN đã chọn bằng validation trên test split.
# Đầu vào: Seed, feature_set, device, test subset/batch và checkpoint tùy chọn.
# Đầu ra: Phase 9 test report cùng binary metrics.
# Lưu ý: Kiểm tra artifact hash và không dùng test để chọn epoch.
def evaluate_baseline_checkpoint(
    *,
    seed: int,
    feature_set: str,
    device_name: str = "auto",
    test_limit: int = 0,
    batch_size: int = 16,
    output_dir: Path = PROCESSED_DATA_DIR,
    report_dir: Path = REPORTS_DIR,
    runs_dir: Path = RUNS_DIR,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    if test_limit < 0 or test_limit == 1:
        raise ValueError("test_limit must be zero or at least two")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_dir = require_project_path(output_dir)
    report_dir = require_project_path(report_dir)
    runs_dir = require_project_path(runs_dir)
    checkpoint_path = _checkpoint_from_report(
        model_name="hgnn",
        seed=seed,
        feature_set=feature_set,
        report_dir=report_dir,
        runs_dir=runs_dir,
        checkpoint_path=checkpoint_path,
    )
    device = resolve_device(device_name)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    validate_checkpoint_artifacts(checkpoint, output_dir)

    saved_config = dict(checkpoint["config"])
    saved_config["device"] = device_name
    config = BaselineConfig(**saved_config)
    config.validate()
    if config.seed != seed or config.feature_set != feature_set:
        raise RuntimeError("Checkpoint identity does not match requested experiment")
    if checkpoint.get("feature_names") != list(feature_columns(feature_set)):
        raise RuntimeError("Checkpoint feature schema does not match feature_set")

    train_graph = load_train_hypergraph(
        seed, output_dir, feature_set=feature_set
    )
    model = HGNNBaseline(
        input_dim=train_graph.features.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    with LocalGraphStore(
        seed,
        "test",
        output_dir,
        feature_set=feature_set,
    ) as test_store:
        test_node_ids, test_labels = select_evaluation_targets(
            test_store,
            output_dir / "nodes.parquet",
            limit=test_limit,
            seed=seed,
        )
        test_metrics = _evaluate_local(
            model,
            test_store,
            test_node_ids,
            test_labels,
            batch_size=batch_size,
            device=device,
        )

    report = {
        "phase": 9,
        "model": "hgnn_baseline",
        "scope": (
            "final baseline evaluation from a validation-selected checkpoint"
            if test_limit == 0
            else "baseline smoke evaluation on a fixed test subset"
        ),
        "generated_utc": utc_now(),
        "seed": seed,
        "feature_set": feature_set,
        "resolved_device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection_split": "validation",
        "checkpoint_validation_metrics": checkpoint["validation_metrics"],
        "configuration_id": configuration_id(config),
        "experiment_id": experiment_id(
            config,
            output_dir / GRAPH_MANIFEST_ARTIFACT,
            output_dir / FEATURE_MANIFEST,
        ),
        "artifact_hashes_verified": True,
        "test_split_used_for_selection": False,
        "test_targets": int(test_node_ids.size),
        "test_sampling": "all" if test_limit == 0 else "fixed random subset",
        "test_metrics": test_metrics,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"phase9_test_hgnn_{feature_set}_seed_{seed}.json"
    write_json_atomic(report_path, report)
    return report


# Mục đích: Đọc family và cardinality theo đúng thứ tự cột của train H0.
# Đầu vào: Processed output directory và seed.
# Đầu ra: Hai NumPy arrays families và sizes.
# Lưu ý: ORDER BY hyperedge_id phải khớp column order lúc materialize matrix.
def load_hyperedge_metadata(
    output_dir: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""SELECT family, size
                FROM read_parquet(
                    '{sql_path(output_dir / HYPEREDGE_METADATA_ARTIFACT)}'
                )
                WHERE seed=? ORDER BY hyperedge_id""",
            [seed],
        ).fetchnumpy()
    finally:
        connection.close()
    return rows["family"].astype(str), rows["size"].astype(np.int64)


# Mục đích: Đánh giá HGSL trên independent local validation/test graphs.
# Đầu vào: Model, graph store, target IDs/labels, batch size, device và seed.
# Đầu ra: Tuple binary metrics và tổng refinement audit.
# Lưu ý: deterministic=True refine mọi local edge nên kết quả không phụ thuộc batch size.
def _evaluate_hgsl(
    model: HGSLModel,
    store: LocalGraphStore,
    target_node_ids: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], dict[str, int]]:
    probabilities: list[np.ndarray] = []
    audit_totals = {
        "sampled_hyperedges": 0,
        "membership_candidates": 0,
        "retained_positive_memberships": 0,
        "added_negative_memberships": 0,
        "restored_isolated_nodes": 0,
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, target_node_ids.size, batch_size):
            batch_ids = target_node_ids[start : start + batch_size]
            batch = batch_local_hypergraphs(store.load_many(batch_ids))
            features = torch.as_tensor(
                batch.features, dtype=torch.float32, device=device
            )
            operator = HypergraphOperator.from_scipy(
                batch.incidence, device=device
            )
            generator = np.random.default_rng(seed * 1_000_003 + start)
            output = model(
                features,
                operator,
                batch.incidence,
                batch.families,
                batch.sizes,
                generator,
                batch.node_graph_ids,
                batch.edge_graph_ids,
                deterministic=True,
            )
            target_indices = torch.as_tensor(
                batch.target_indices, dtype=torch.int64, device=device
            )
            target_probabilities = torch.sigmoid(output.logits[target_indices])
            probabilities.append(target_probabilities.cpu().numpy())
            for key in audit_totals:
                audit_totals[key] += int(output.refinement.audit[key])
    metrics = binary_metrics(labels, np.concatenate(probabilities))
    return metrics, audit_totals


# Mục đích: Lưu atomic HGSL checkpoint tốt nhất theo validation AUC.
# Đầu vào: Path, model/optimizer, config, epoch, metrics, audit và manifests.
# Đầu ra: Không trả dữ liệu; tạo checkpoint .pt đầy đủ state.
# Lưu ý: Lưu cả RNG state để có thể truy vết run.
def _write_hgsl_checkpoint(
    path: Path,
    model: HGSLModel,
    optimizer: torch.optim.Optimizer,
    config: HGSLTrainingConfig,
    epoch: int,
    validation_metrics: dict[str, float],
    refinement_audit: dict[str, Any],
    graph_manifest_path: Path,
    feature_manifest_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    checkpoint = {
        "phase": 8,
        "epoch": epoch,
        "config": asdict(config),
        "validation_metrics": validation_metrics,
        "refinement_audit": refinement_audit,
        "graph_manifest": source_signature(
            graph_manifest_path, include_hash=True
        ),
        "feature_manifest": source_signature(
            feature_manifest_path, include_hash=True
        ),
        "feature_names": list(feature_columns(config.feature_set)),
        "model_state": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(checkpoint, temporary)
    temporary.replace(path)


# Mục đích: Train HGSL end-to-end và early-stop theo validation AUC.
# Đầu vào: HGSLTrainingConfig cùng output/report/runs directories.
# Đầu ra: Report history, best epoch/checkpoint và manifest hashes.
# Lưu ý: Train dùng sampled H*; validation dùng deterministic local H*.
def run_hgsl_training(
    config: HGSLTrainingConfig,
    output_dir: Path = PROCESSED_DATA_DIR,
    report_dir: Path = REPORTS_DIR,
    runs_dir: Path = RUNS_DIR,
) -> dict[str, Any]:
    config.validate()
    output_dir = require_project_path(output_dir)
    report_dir = require_project_path(report_dir)
    runs_dir = require_project_path(runs_dir)
    set_seed(config.seed)
    device = resolve_device(config.device)

    graph = load_train_hypergraph(
        config.seed, output_dir, feature_set=config.feature_set
    )
    families, sizes = load_hyperedge_metadata(output_dir, config.seed)
    features = torch.as_tensor(
        graph.features, dtype=torch.float32, device=device
    )
    labels_numpy = labels_for_nodes(output_dir / "nodes.parquet", graph.node_ids)
    labels = torch.as_tensor(labels_numpy, dtype=torch.float32, device=device)
    positive_weight = train_pos_weight(labels)
    initial_operator = HypergraphOperator.from_scipy(
        graph.incidence, device=device
    )
    model = HGSLModel(
        input_dim=features.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        refinement=config.refinement_config(),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    graph_manifest_path = output_dir / GRAPH_MANIFEST_ARTIFACT
    feature_manifest_path = output_dir / FEATURE_MANIFEST
    run_id = experiment_id(config, graph_manifest_path, feature_manifest_path)
    checkpoint_path = runs_dir / (
        f"hgsl_{config.feature_set}_seed_{config.seed}_{run_id}.pt"
    )
    history: list[dict[str, Any]] = []
    best_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    with LocalGraphStore(
        config.seed,
        "validation",
        output_dir,
        feature_set=config.feature_set,
    ) as validation_store:
        validation_node_ids, validation_labels = select_evaluation_targets(
            validation_store,
            output_dir / "nodes.parquet",
            limit=config.validation_limit,
            seed=config.seed,
        )

        for epoch in range(1, config.epochs + 1):
            generator = np.random.default_rng(config.seed * 100_003 + epoch)
            contrastive_count = min(
                config.contrastive_nodes, graph.incidence.shape[0]
            )
            sampled_nodes = generator.choice(
                graph.incidence.shape[0],
                size=contrastive_count,
                replace=False,
            )
            contrastive_indices = torch.as_tensor(
                sampled_nodes, dtype=torch.int64, device=device
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(
                features,
                initial_operator,
                graph.incidence,
                families,
                sizes,
                generator,
            )
            total_loss, loss_parts = hgsl_objective(
                output.logits,
                labels,
                output.z0,
                output.z_star,
                positive_weight,
                contrastive_weight=config.contrastive_weight,
                temperature=config.contrastive_temperature,
                contrastive_node_indices=contrastive_indices,
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError("HGSL loss is non-finite")
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_gradient_norm,
                error_if_nonfinite=True,
            )
            scorer_terms = [
                torch.sum(parameter.grad.square())
                for parameter in model.refiner.scorer.parameters()
                if parameter.grad is not None
            ]
            if not scorer_terms:
                raise RuntimeError("No gradient reached the membership scorer")
            scorer_gradient_norm = torch.sqrt(torch.stack(scorer_terms).sum())
            if not torch.isfinite(scorer_gradient_norm) or scorer_gradient_norm <= 0:
                raise RuntimeError("Membership scorer gradient is invalid")
            optimizer.step()

            validation_metrics, validation_refinement = _evaluate_hgsl(
                model,
                validation_store,
                validation_node_ids,
                validation_labels,
                batch_size=config.validation_batch_size,
                device=device,
                seed=config.seed,
            )
            epoch_record = {
                "epoch": epoch,
                "loss": {
                    "total": float(total_loss.detach().cpu()),
                    "bce": float(loss_parts["bce"].detach().cpu()),
                    "contrastive": float(
                        loss_parts["contrastive"].detach().cpu()
                    ),
                },
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "scorer_gradient_norm": float(
                    scorer_gradient_norm.detach().cpu()
                ),
                "train_refinement": output.refinement.audit,
                "validation_metrics": validation_metrics,
                "validation_refinement": validation_refinement,
            }
            history.append(epoch_record)

            if validation_metrics["auc"] > best_auc:
                best_auc = validation_metrics["auc"]
                best_epoch = epoch
                epochs_without_improvement = 0
                _write_hgsl_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    config,
                    epoch,
                    validation_metrics,
                    output.refinement.audit,
                    graph_manifest_path,
                    feature_manifest_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break

    graph_manifest = source_signature(graph_manifest_path, include_hash=True)
    feature_manifest = source_signature(feature_manifest_path, include_hash=True)
    report = {
        "phase": 8,
        "scope": (
            "HGSL training with complete validation"
            if config.validation_limit == 0
            else "HGSL smoke training with sampled validation"
        ),
        "generated_utc": utc_now(),
        "configuration_id": configuration_id(config),
        "experiment_id": run_id,
        "torch_version": torch.__version__,
        "config": asdict(config),
        "resolved_device": str(device),
        "nodes": int(graph.incidence.shape[0]),
        "hyperedges": int(graph.incidence.shape[1]),
        "feature_dim": int(features.shape[1]),
        "feature_names": list(feature_columns(config.feature_set)),
        "pos_weight": float(positive_weight.detach().cpu()),
        "validation_targets": int(validation_node_ids.size),
        "validation_sampling": (
            "all" if config.validation_limit == 0 else "fixed random subset"
        ),
        "selection_metric": "validation_auc",
        "test_split_used": False,
        "local_negative_scope": "same local graph only",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_metrics": history[best_epoch - 1]["validation_metrics"],
        "history": history,
        "checkpoint": str(checkpoint_path),
        "graph_manifest_sha256": graph_manifest["sha256"],
        "feature_manifest_sha256": feature_manifest["sha256"],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    latest_report_path = (
        report_dir / f"phase8_hgsl_{config.feature_set}_seed_{config.seed}.json"
    )
    detailed_report_path = report_dir / (
        f"phase8_hgsl_{config.feature_set}_seed_{config.seed}_{run_id}.json"
    )
    write_json_atomic(latest_report_path, report)
    write_json_atomic(detailed_report_path, report)
    return report


# Mục đích: Chạy một optimizer step để kiểm tra H0 -> Z0 -> H* -> Z*.
# Đầu vào: HGSLTrainingConfig và artifact/report directories.
# Đầu ra: Phase 7 smoke report gồm loss, gradient và refinement audit.
# Lưu ý: Không dùng kết quả này làm kết quả nghiên cứu hoặc chọn hyperparameter.
def run_hgsl_smoke(
    config: HGSLTrainingConfig,
    output_dir: Path = PROCESSED_DATA_DIR,
    report_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    config.validate()
    output_dir = require_project_path(output_dir)
    report_dir = require_project_path(report_dir)
    set_seed(config.seed)
    device = resolve_device(config.device)
    graph = load_train_hypergraph(
        config.seed, output_dir, feature_set=config.feature_set
    )
    families, sizes = load_hyperedge_metadata(output_dir, config.seed)
    if families.size != graph.incidence.shape[1]:
        raise RuntimeError("Hyperedge metadata does not match H0 columns")

    features = torch.as_tensor(
        graph.features, dtype=torch.float32, device=device
    )
    labels_numpy = labels_for_nodes(output_dir / "nodes.parquet", graph.node_ids)
    labels = torch.as_tensor(labels_numpy, dtype=torch.float32, device=device)
    initial_operator = HypergraphOperator.from_scipy(
        graph.incidence, device=device
    )
    model = HGSLModel(
        input_dim=features.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        refinement=config.refinement_config(),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = np.random.default_rng(config.seed)
    contrastive_count = min(
        config.contrastive_nodes, graph.incidence.shape[0]
    )
    sampled_nodes = generator.choice(
        graph.incidence.shape[0],
        size=contrastive_count,
        replace=False,
    )
    contrastive_indices = torch.as_tensor(
        sampled_nodes, dtype=torch.int64, device=device
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        features,
        initial_operator,
        graph.incidence,
        families,
        sizes,
        generator,
    )
    total_loss, loss_parts = hgsl_objective(
        output.logits,
        labels,
        output.z0,
        output.z_star,
        train_pos_weight(labels),
        contrastive_weight=config.contrastive_weight,
        temperature=config.contrastive_temperature,
        contrastive_node_indices=contrastive_indices,
    )
    if not torch.isfinite(total_loss):
        raise RuntimeError("HGSL loss is non-finite")
    total_loss.backward()

    squared_norm = torch.zeros((), dtype=torch.float32, device=device)
    scorer_squared_norm = torch.zeros((), dtype=torch.float32, device=device)
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing or non-finite gradient: {name}")
        parameter_norm = torch.sum(parameter.grad.square())
        squared_norm += parameter_norm
        if name.startswith("refiner.scorer"):
            scorer_squared_norm += parameter_norm
    gradient_norm = torch.sqrt(squared_norm)
    scorer_gradient_norm = torch.sqrt(scorer_squared_norm)
    if scorer_gradient_norm <= 0:
        raise RuntimeError("No gradient reached the membership scorer")
    optimizer.step()

    report = {
        "phase": 7,
        "scope": "one-step HGSL smoke run; hyperparameters are not selected",
        "generated_utc": utc_now(),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "resolved_device": str(device),
        "nodes": int(graph.incidence.shape[0]),
        "hyperedges": int(graph.incidence.shape[1]),
        "feature_dim": int(features.shape[1]),
        "feature_names": list(feature_columns(config.feature_set)),
        "loss": {
            "total": float(total_loss.detach().cpu()),
            "bce": float(loss_parts["bce"].detach().cpu()),
            "contrastive": float(loss_parts["contrastive"].detach().cpu()),
        },
        "gradient_finite": True,
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "scorer_gradient_norm": float(scorer_gradient_norm.detach().cpu()),
        "z0_shape": list(output.z0.shape),
        "z_star_shape": list(output.z_star.shape),
        "refinement": output.refinement.audit,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        report_dir / f"phase7_hgsl_{config.feature_set}_seed_{config.seed}.json"
    )
    write_json_atomic(report_path, report)
    return report


# Mục đích: Đánh giá checkpoint HGSL validation-selected trên test split.
# Đầu vào: Seed, feature_set, device, test protocol, artifact dirs và checkpoint.
# Đầu ra: Phase 9 report có test metrics và refinement audit.
# Lưu ý: Xác minh identity/schema/hash trước khi đọc test predictions.
def evaluate_hgsl_checkpoint(
    *,
    seed: int,
    feature_set: str,
    device_name: str = "auto",
    test_limit: int = 0,
    batch_size: int = 4,
    output_dir: Path = PROCESSED_DATA_DIR,
    report_dir: Path = REPORTS_DIR,
    runs_dir: Path = RUNS_DIR,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    if test_limit < 0 or test_limit == 1:
        raise ValueError("test_limit must be zero or at least two")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_dir = require_project_path(output_dir)
    report_dir = require_project_path(report_dir)
    runs_dir = require_project_path(runs_dir)
    checkpoint_path = _checkpoint_from_report(
        model_name="hgsl",
        seed=seed,
        feature_set=feature_set,
        report_dir=report_dir,
        runs_dir=runs_dir,
        checkpoint_path=checkpoint_path,
    )

    device = resolve_device(device_name)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    validate_checkpoint_artifacts(checkpoint, output_dir)
    saved_config = dict(checkpoint["config"])
    saved_config["device"] = device_name
    config = HGSLTrainingConfig(**saved_config)
    config.validate()
    if config.seed != seed or config.feature_set != feature_set:
        raise RuntimeError("Checkpoint identity does not match requested experiment")
    if checkpoint.get("feature_names") != list(feature_columns(feature_set)):
        raise RuntimeError("Checkpoint feature schema does not match feature_set")

    train_graph = load_train_hypergraph(
        seed, output_dir, feature_set=feature_set
    )
    model = HGSLModel(
        input_dim=train_graph.features.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        refinement=config.refinement_config(),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    with LocalGraphStore(
        seed,
        "test",
        output_dir,
        feature_set=feature_set,
    ) as test_store:
        test_node_ids, test_labels = select_evaluation_targets(
            test_store,
            output_dir / "nodes.parquet",
            limit=test_limit,
            seed=seed,
        )
        test_metrics, refinement_audit = _evaluate_hgsl(
            model,
            test_store,
            test_node_ids,
            test_labels,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )

    report = {
        "phase": 9,
        "scope": (
            "final evaluation of a validation-selected checkpoint"
            if test_limit == 0
            else "smoke evaluation on a fixed test subset"
        ),
        "generated_utc": utc_now(),
        "seed": seed,
        "feature_set": feature_set,
        "resolved_device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection_split": "validation",
        "checkpoint_validation_metrics": checkpoint["validation_metrics"],
        "configuration_id": configuration_id(config),
        "experiment_id": experiment_id(
            config,
            output_dir / GRAPH_MANIFEST_ARTIFACT,
            output_dir / FEATURE_MANIFEST,
        ),
        "artifact_hashes_verified": True,
        "inference_refinement": "all local hyperedges; deterministic node candidates",
        "test_split_used_for_selection": False,
        "test_targets": int(test_node_ids.size),
        "test_sampling": "all" if test_limit == 0 else "fixed random subset",
        "test_metrics": test_metrics,
        "test_refinement": refinement_audit,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"phase9_test_{feature_set}_seed_{seed}.json"
    write_json_atomic(report_path, report)
    return report
