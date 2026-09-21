# Học lại cấu trúc hypergraph theo đúng pipeline: sample edge, sample node,
# chấm điểm membership, dựng sparse H* và tính classification/contrastive loss.
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional

from model import HGNNBaseline, HypergraphOperator


SIZE_BUCKETS = ("small", "medium", "large")


# Mục đích: Chia hyperedge thành small/medium/large để sampling cân bằng.
# Đầu vào: Vector cardinality của hyperedges.
# Đầu ra: Vector chuỗi bucket cùng số phần tử.
# Lưu ý: Mọi edge phải có ít nhất hai node.
def cardinality_buckets(sizes: np.ndarray) -> np.ndarray:
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    if sizes.size == 0 or np.any(sizes < 2):
        raise ValueError("sizes must contain hyperedge cardinalities >= 2")
    return np.where(
        sizes <= 10,
        "small",
        np.where(sizes <= 100, "medium", "large"),
    )


# Mục đích: Sample round-robin giữa các family và size bucket.
# Đầu vào: Edge families, sizes, sample budget và NumPy generator.
# Đầu ra: Vector hyperedge IDs được chọn.
# Lưu ý: Tránh Behavioral hoặc edge lớn chiếm toàn bộ sample.
def sample_balanced_hyperedges(
    families: np.ndarray,
    sizes: np.ndarray,
    sample_size: int,
    generator: np.random.Generator,
) -> np.ndarray:
    families = np.asarray(families).astype(str).reshape(-1)
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    if families.shape != sizes.shape or families.size == 0:
        raise ValueError("families and sizes must be non-empty with equal shape")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    buckets = cardinality_buckets(sizes)
    strata: dict[tuple[str, str], np.ndarray] = {}
    pairs = set(zip(families, buckets, strict=True))
    for family, bucket in sorted(pairs):
        in_stratum = (families == family) & (buckets == bucket)
        indices = np.flatnonzero(in_stratum)
        strata[(family, bucket)] = generator.permutation(indices)

    target_count = min(sample_size, families.size)
    cursors = {key: 0 for key in strata}
    selected: list[int] = []
    while len(selected) < target_count:
        found_edge = False
        for key, indices in strata.items():
            cursor = cursors[key]
            if cursor >= indices.size:
                continue
            selected.append(int(indices[cursor]))
            cursors[key] += 1
            found_edge = True
            if len(selected) == target_count:
                break
        if not found_edge:
            break
    return np.asarray(selected, dtype=np.int64)


# Mục đích: Đếm sampled edges theo family:size bucket để ghi audit.
# Đầu vào: Edge IDs đã sample cùng family/size arrays đầy đủ.
# Đầu ra: Dictionary số lượng của từng stratum.
def sampled_strata_audit(
    edge_ids: np.ndarray,
    families: np.ndarray,
    sizes: np.ndarray,
) -> dict[str, int]:
    buckets = cardinality_buckets(np.asarray(sizes)[edge_ids])
    selected_families = np.asarray(families).astype(str)[edge_ids]
    counts = Counter(
        f"{family}:{bucket}"
        for family, bucket in zip(selected_families, buckets, strict=True)
    )
    return dict(sorted(counts.items()))


# Mục đích: Gom positive/negative node candidates theo từng sampled edge.
# Đầu vào: Edge IDs và tuple node arrays tương ứng.
# Đầu ra: Object bất biến dùng bởi HypergraphRefiner.
@dataclass(frozen=True)
class SampledMemberships:
    edge_ids: np.ndarray
    positive_nodes: tuple[np.ndarray, ...]
    negative_nodes: tuple[np.ndarray, ...]


# Mục đích: Chọn node không thuộc một hyperedge làm negative candidates.
# Đầu vào: Số node, incident nodes, số lượng, RNG, phạm vi node và mode deterministic.
# Đầu ra: Vector negative node IDs đã sort.
# Lưu ý: allowed_nodes giới hạn candidate trong cùng local graph khi evaluation.
def _sample_negative_nodes(
    node_count: int,
    incident_nodes: np.ndarray,
    count: int,
    generator: np.random.Generator,
    allowed_nodes: np.ndarray | None = None,
    deterministic: bool = False,
) -> np.ndarray:
    if allowed_nodes is not None:
        candidates = np.setdiff1d(
            np.asarray(allowed_nodes, dtype=np.int64),
            incident_nodes,
            assume_unique=False,
        )
        target_count = min(count, candidates.size)
        if target_count == 0:
            return np.empty(0, dtype=np.int64)
        if deterministic:
            return np.sort(candidates[:target_count])
        selected = generator.choice(
            candidates, size=target_count, replace=False
        )
        return np.sort(selected)

    available_count = node_count - incident_nodes.size
    target_count = min(count, available_count)
    if target_count == 0:
        return np.empty(0, dtype=np.int64)

    if deterministic:
        candidates = np.setdiff1d(
            np.arange(node_count, dtype=np.int64),
            incident_nodes,
            assume_unique=False,
        )
        return candidates[:target_count]

    incident_set = set(int(node) for node in incident_nodes)
    selected_set: set[int] = set()
    while len(selected_set) < target_count:
        remaining = target_count - len(selected_set)
        draws = generator.integers(0, node_count, size=max(2 * remaining, 8))
        for node in draws:
            node_id = int(node)
            if node_id not in incident_set:
                selected_set.add(node_id)

    selected = np.fromiter(selected_set, dtype=np.int64)
    if selected.size > target_count:
        selected = generator.choice(selected, size=target_count, replace=False)
    return np.sort(selected)


# Mục đích: Sample positive incident và negative non-incident nodes cho từng edge.
# Đầu vào: Incidence, edge IDs, budgets, RNG và group arrays tùy chọn.
# Đầu ra: SampledMemberships không có overlap positive/negative.
# Lưu ý: deterministic=True lấy candidate theo thứ tự cố định cho validation/test.
def sample_incident_nodes(
    incidence: sparse.spmatrix,
    edge_ids: np.ndarray,
    *,
    positive_count: int,
    negative_count: int,
    generator: np.random.Generator,
    node_groups: np.ndarray | None = None,
    edge_groups: np.ndarray | None = None,
    deterministic: bool = False,
) -> SampledMemberships:
    if positive_count <= 0 or negative_count < 0:
        raise ValueError("positive_count must be positive and negative_count non-negative")
    matrix = incidence.tocsc()
    edge_ids = np.asarray(edge_ids, dtype=np.int64).reshape(-1)
    invalid_edge = np.any(edge_ids < 0) or np.any(edge_ids >= matrix.shape[1])
    if edge_ids.size == 0 or invalid_edge:
        raise ValueError("edge_ids contain an invalid hyperedge")
    if (node_groups is None) != (edge_groups is None):
        raise ValueError("node_groups and edge_groups must be provided together")
    if node_groups is not None:
        node_groups = np.asarray(node_groups).reshape(-1)
        edge_groups = np.asarray(edge_groups).reshape(-1)
        if node_groups.size != matrix.shape[0] or edge_groups.size != matrix.shape[1]:
            raise ValueError("Group arrays must match incidence dimensions")

    positive_samples: list[np.ndarray] = []
    negative_samples: list[np.ndarray] = []
    for edge_id in edge_ids:
        start, stop = matrix.indptr[edge_id : edge_id + 2]
        incident_nodes = matrix.indices[start:stop].astype(np.int64, copy=False)
        if incident_nodes.size == 0:
            raise ValueError("Cannot sample an empty hyperedge")

        selected_positive_count = min(positive_count, incident_nodes.size)
        if deterministic:
            positive_nodes = np.sort(incident_nodes[:selected_positive_count])
        else:
            positive_nodes = generator.choice(
                incident_nodes, size=selected_positive_count, replace=False
            )
            positive_nodes = np.sort(positive_nodes)

        allowed_nodes = None
        if node_groups is not None and edge_groups is not None:
            allowed_nodes = np.flatnonzero(node_groups == edge_groups[edge_id])
        negative_nodes = _sample_negative_nodes(
            matrix.shape[0],
            incident_nodes,
            negative_count,
            generator,
            allowed_nodes,
            deterministic,
        )
        if np.intersect1d(positive_nodes, negative_nodes).size:
            raise RuntimeError("Positive and negative samples overlap")
        positive_samples.append(positive_nodes)
        negative_samples.append(negative_nodes)

    return SampledMemberships(
        edge_ids,
        tuple(positive_samples),
        tuple(negative_samples),
    )


# Mục đích: Lưu các hyperparameter trực tiếp của bước refine H*.
# Đầu vào: Sampling budgets, selection mode, top_r và threshold.
# Đầu ra: Config bất biến được truyền vào HypergraphRefiner.
@dataclass(frozen=True)
class RefinementConfig:
    sampled_hyperedges: int = 96
    positive_nodes: int = 16
    negative_nodes: int = 16
    mode: str = "top_r"
    top_r: int = 8
    threshold: float = 0.5

    # Mục đích: Kiểm tra mọi sampling/selection hyperparameter hợp lệ.
    # Đầu vào: Các field của config hiện tại.
    # Đầu ra: Không có nếu hợp lệ; ném ValueError nếu sai.
    def validate(self) -> None:
        if self.sampled_hyperedges <= 0 or self.positive_nodes <= 0:
            raise ValueError("Sampling counts must be positive")
        if self.negative_nodes < 0 or self.top_r <= 0:
            raise ValueError("negative_nodes must be non-negative and top_r positive")
        if self.mode not in {"top_r", "threshold"}:
            raise ValueError("mode must be top_r or threshold")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between zero and one")


# Mục đích: Gom output của bước refine để HGNN thứ hai và report cùng sử dụng.
# Đầu vào: Operator H*, sparse incidence, sampled edge IDs và audit.
# Đầu ra: Object bất biến mô tả refined hypergraph của forward hiện tại.
@dataclass(frozen=True)
class RefinementResult:
    operator: HypergraphOperator
    incidence: torch.Tensor
    sampled_edge_ids: np.ndarray
    audit: dict[str, Any]


# Mục đích: Học compatibility score giữa một node và một hyperedge embedding.
# Đầu vào: Embedding dimension khi khởi tạo; hai embedding batches khi forward.
# Đầu ra: Một membership logit cho mỗi node-edge candidate pair.
class MembershipScorer(nn.Module):
    # Mục đích: Khởi tạo hai projection matrices và scalar bias.
    # Đầu vào: Embedding dimension.
    # Đầu ra: Learnable scorer parameters.
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.node_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.edge_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = sqrt(embedding_dim)

    # Mục đích: Tính bilinear-like compatibility score đã scale.
    # Đầu vào: Node embeddings và edge embeddings có cùng shape [C,D].
    # Đầu ra: Logit vector [C].
    def forward(
        self, node_embeddings: torch.Tensor, edge_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if node_embeddings.shape != edge_embeddings.shape:
            raise ValueError("Node and edge embedding batches must have equal shape")
        node_projection = self.node_projection(node_embeddings)
        edge_projection = self.edge_projection(edge_embeddings)
        return (node_projection * edge_projection).sum(dim=1) / self.scale + self.bias


# Mục đích: Refine candidate memberships và tạo sparse weighted H*.
# Đầu vào: Z0, H0, edge metadata, RNG và group information tùy chọn.
# Đầu ra: RefinementResult chứa operator/incidence/audit của H*.
# Lưu ý: Train sample edge; deterministic validation/test refine toàn bộ local edge.
class HypergraphRefiner(nn.Module):
    # Mục đích: Lưu config và tạo membership scorer.
    # Đầu vào: Embedding dimension và RefinementConfig.
    # Đầu ra: Refiner có learnable scorer.
    def __init__(self, embedding_dim: int, config: RefinementConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.scorer = MembershipScorer(embedding_dim)

    # Mục đích: Chọn candidate membership bằng top-r hoặc threshold.
    # Đầu vào: Candidate logits và số positive candidates ở đầu vector.
    # Đầu ra: Tensor indices của candidates được giữ.
    # Lưu ý: Luôn giữ ít nhất một positive để edge không rỗng.
    def _select(
        self, logits: torch.Tensor, positive_count: int
    ) -> torch.Tensor:
        if self.config.mode == "top_r":
            count = min(self.config.top_r, logits.numel())
            selected = torch.topk(logits, count).indices
        else:
            selected = torch.flatnonzero(
                torch.sigmoid(logits) >= self.config.threshold
            )
            if selected.numel() == 0:
                selected = torch.argmax(logits).reshape(1)
        if not torch.any(selected < positive_count):
            best_positive = torch.argmax(logits[:positive_count])
            if self.config.mode == "top_r" and selected.numel() >= self.config.top_r:
                selected_logits = logits[selected]
                selected = selected.clone()
                selected[torch.argmin(selected_logits)] = best_positive
            else:
                selected = torch.cat((selected, best_positive.reshape(1)))
        return torch.unique(selected)

    # Mục đích: Chạy toàn bộ sampling, scoring, selection và dựng H*.
    # Đầu vào: Embeddings, SciPy H0, family/size arrays, RNG và group IDs.
    # Đầu ra: RefinementResult với sparse differentiable membership weights.
    # Lưu ý: Node có nguy cơ cô lập được phục hồi một membership cũ.
    def forward(
        self,
        embeddings: torch.Tensor,
        incidence: sparse.spmatrix,
        families: np.ndarray,
        sizes: np.ndarray,
        generator: np.random.Generator,
        node_groups: np.ndarray | None = None,
        edge_groups: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> RefinementResult:
        matrix = incidence.tocsr()
        families = np.asarray(families).astype(str).reshape(-1)
        sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
        if embeddings.ndim != 2 or embeddings.shape[0] != matrix.shape[0]:
            raise ValueError("Embedding rows must match incidence rows")
        if families.size != matrix.shape[1] or sizes.size != matrix.shape[1]:
            raise ValueError("Hyperedge metadata must match incidence columns")

        if deterministic:
            sampled_edges = np.arange(matrix.shape[1], dtype=np.int64)
        else:
            sampled_edges = sample_balanced_hyperedges(
                families, sizes, self.config.sampled_hyperedges, generator
            )
        sampled = sample_incident_nodes(
            matrix,
            sampled_edges,
            positive_count=self.config.positive_nodes,
            negative_count=self.config.negative_nodes,
            generator=generator,
            node_groups=node_groups,
            edge_groups=edge_groups,
            deterministic=deterministic,
        )
        positive_tensors = [
            torch.as_tensor(nodes, dtype=torch.int64, device=embeddings.device)
            for nodes in sampled.positive_nodes
        ]
        edge_embeddings = torch.stack(
            [embeddings[nodes].mean(dim=0) for nodes in positive_tensors]
        )

        selected_nodes: list[np.ndarray] = []
        selected_edges: list[np.ndarray] = []
        selected_values: list[torch.Tensor] = []
        retained_positive = 0
        added_negative = 0
        candidate_count = 0
        for position, edge_id in enumerate(sampled.edge_ids):
            positive = sampled.positive_nodes[position]
            negative = sampled.negative_nodes[position]
            candidates = np.concatenate((positive, negative))
            candidate_count += int(candidates.size)
            candidate_tensor = torch.as_tensor(
                candidates, dtype=torch.int64, device=embeddings.device
            )
            repeated_edge = edge_embeddings[position].expand(candidates.size, -1)
            logits = self.scorer(embeddings[candidate_tensor], repeated_edge)
            selected = self._select(logits, positive.size)
            selected_numpy = selected.detach().cpu().numpy()
            selected_nodes.append(candidates[selected_numpy])
            selected_edges.append(
                np.full(selected.numel(), edge_id, dtype=np.int64)
            )
            selected_values.extend(torch.sigmoid(logits[selected]).unbind())
            retained_positive += int(np.count_nonzero(selected_numpy < positive.size))
            added_negative += int(np.count_nonzero(selected_numpy >= positive.size))

        coo = matrix.tocoo(copy=False)
        keep = ~np.isin(coo.col, sampled_edges)
        rows = [coo.row[keep].astype(np.int64, copy=False), *selected_nodes]
        columns = [coo.col[keep].astype(np.int64, copy=False), *selected_edges]
        combined_rows = np.concatenate(rows)
        combined_columns = np.concatenate(columns)

        row_counts = np.bincount(combined_rows, minlength=matrix.shape[0])
        missing_nodes = np.flatnonzero(row_counts == 0)
        fallback_edges = np.empty(missing_nodes.size, dtype=np.int64)
        for index, node in enumerate(missing_nodes):
            start = matrix.indptr[node]
            fallback_edges[index] = matrix.indices[start]
        if missing_nodes.size:
            combined_rows = np.concatenate((combined_rows, missing_nodes))
            combined_columns = np.concatenate((combined_columns, fallback_edges))

        constant_count = int(np.count_nonzero(keep))
        device = embeddings.device
        values = torch.cat(
            (
                torch.ones(constant_count, dtype=embeddings.dtype, device=device),
                torch.stack(selected_values),
                torch.ones(missing_nodes.size, dtype=embeddings.dtype, device=device),
            )
        )
        indices = torch.as_tensor(
            np.vstack((combined_rows, combined_columns)),
            dtype=torch.int64,
            device=device,
        )
        refined_incidence = torch.sparse_coo_tensor(
            indices,
            values,
            size=matrix.shape,
            dtype=embeddings.dtype,
            device=device,
            check_invariants=False,
        ).coalesce()
        operator = HypergraphOperator.from_torch_sparse(refined_incidence)
        audit = {
            "sampled_hyperedges": int(sampled_edges.size),
            "sampled_strata": sampled_strata_audit(
                sampled_edges, families, sizes
            ),
            "membership_candidates": candidate_count,
            "retained_positive_memberships": retained_positive,
            "added_negative_memberships": added_negative,
            "restored_isolated_nodes": int(missing_nodes.size),
            "initial_incidences": int(matrix.nnz),
            "refined_incidences": int(refined_incidence._nnz()),
            "mean_learned_membership": float(
                torch.stack(selected_values).mean().detach().cpu()
            ),
            "mode": self.config.mode,
            "top_r": self.config.top_r,
            "threshold": self.config.threshold,
            "deterministic": deterministic,
        }
        return RefinementResult(operator, refined_incidence, sampled_edges, audit)


# Mục đích: Gom những tensor cần cho prediction và hai thành phần loss.
# Đầu vào: Logits, Z0, Z* và RefinementResult.
# Đầu ra: Object bất biến trả về từ HGSLModel.forward().
@dataclass(frozen=True)
class HGSLForward:
    logits: torch.Tensor
    z0: torch.Tensor
    z_star: torch.Tensor
    refinement: RefinementResult


# Mục đích: Ghép hai lần HGNN với bước học cấu trúc H0 -> H* ở giữa.
# Đầu vào: Model dimensions và refinement config khi khởi tạo.
# Đầu ra: Model tạo Z0, H*, Z* và classification logits.
# Lưu ý: Hai lượt HGNN dùng chung backbone weights.
class HGSLModel(nn.Module):
    # Mục đích: Khởi tạo shared HGNN backbone và HypergraphRefiner.
    # Đầu vào: input_dim, hidden_dim, dropout và refinement config.
    # Đầu ra: HGSL model có thể train end-to-end.
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        refinement: RefinementConfig = RefinementConfig(),
    ) -> None:
        super().__init__()
        self.backbone = HGNNBaseline(input_dim, hidden_dim, dropout)
        self.refiner = HypergraphRefiner(hidden_dim, refinement)

    # Mục đích: Chạy pipeline X+H0 -> Z0 -> H* -> Z* -> logits.
    # Đầu vào: Features, H0 operator/incidence, edge metadata, RNG và group IDs.
    # Đầu ra: HGSLForward.
    # Lưu ý: deterministic=True chỉ dùng ở validation/test để cố định H*.
    def forward(
        self,
        node_features: torch.Tensor,
        initial_operator: HypergraphOperator,
        initial_incidence: sparse.spmatrix,
        families: np.ndarray,
        sizes: np.ndarray,
        generator: np.random.Generator,
        node_groups: np.ndarray | None = None,
        edge_groups: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> HGSLForward:
        z0 = self.backbone.encode(node_features, initial_operator)
        refinement = self.refiner(
            z0,
            initial_incidence,
            families,
            sizes,
            generator,
            node_groups,
            edge_groups,
            deterministic,
        )
        z_star = self.backbone.encode(node_features, refinement.operator)
        logits = self.backbone.classify(z_star)
        return HGSLForward(logits, z0, z_star, refinement)


# Mục đích: Tính BCE pos_weight từ phân bố label train.
# Đầu vào: Binary label tensor không rỗng.
# Đầu ra: Scalar negatives/positives.
# Lưu ý: Không được truyền validation/test labels vào hàm này.
def train_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float().reshape(-1)
    is_binary = torch.all((labels == 0) | (labels == 1))
    if labels.numel() == 0 or not is_binary:
        raise ValueError("labels must be a non-empty binary tensor")
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("Both label classes are required")
    return negatives / positives


# Mục đích: Tính weighted binary cross-entropy cho dropout logits.
# Đầu vào: Logits, labels cùng shape và positive class weight.
# Đầu ra: Scalar BCEWithLogits loss.
def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    logits = logits.reshape(-1)
    labels = labels.float().reshape(-1)
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have the same shape")
    return functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=positive_weight,
    )


# Mục đích: Căn chỉnh Z0 và Z* của cùng node bằng symmetric InfoNCE.
# Đầu vào: Hai embedding matrices, temperature và node subset tùy chọn.
# Đầu ra: Scalar contrastive loss.
# Lưu ý: Subset tránh tạo similarity matrix của toàn bộ 144k train nodes.
def contrastive_alignment_loss(
    z0: torch.Tensor,
    z_star: torch.Tensor,
    *,
    temperature: float = 0.2,
    node_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if z0.shape != z_star.shape or z0.ndim != 2:
        raise ValueError("z0 and z_star must be equal-shape 2-D tensors")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if node_indices is not None:
        z0 = z0[node_indices]
        z_star = z_star[node_indices]
    if z0.shape[0] < 2:
        raise ValueError("At least two nodes are required for contrastive loss")

    initial_view = functional.normalize(z0, dim=1)
    refined_view = functional.normalize(z_star, dim=1)
    similarities = initial_view @ refined_view.T / temperature
    targets = torch.arange(z0.shape[0], device=z0.device)
    forward_loss = functional.cross_entropy(similarities, targets)
    reverse_loss = functional.cross_entropy(similarities.T, targets)
    return 0.5 * (forward_loss + reverse_loss)


# Mục đích: Ghép BCE và weighted contrastive loss thành objective cuối.
# Đầu vào: Prediction tensors, labels, embeddings, pos_weight và lambda/temperature.
# Đầu ra: Total loss và dictionary hai thành phần BCE/contrastive.
# Lưu ý: contrastive_weight=0 trả đúng BCE baseline và bỏ qua InfoNCE.
def hgsl_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    z0: torch.Tensor,
    z_star: torch.Tensor,
    positive_weight: torch.Tensor,
    *,
    contrastive_weight: float,
    temperature: float = 0.2,
    contrastive_node_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if contrastive_weight < 0:
        raise ValueError("contrastive_weight must be non-negative")
    bce = classification_loss(logits, labels, positive_weight)
    if contrastive_weight == 0:
        contrastive = bce.new_zeros(())
        total = bce
    else:
        contrastive = contrastive_alignment_loss(
            z0,
            z_star,
            temperature=temperature,
            node_indices=contrastive_node_indices,
        )
        total = bce + contrastive_weight * contrastive
    return total, {"bce": bce, "contrastive": contrastive}
