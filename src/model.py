# Cài đặt phần neural network HGNN và phép lan truyền hypergraph sparse.
# Việc dựng H0 và học lại H* nằm ở module khác để file này chỉ giữ toán của HGNN.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
import torch
from torch import nn


SPARSE_BACKWARD_CHUNK = 100_000


# Mục đích: Nhân sparse-dense nhưng vẫn tính gradient cho sparse values với O(nnz).
# Đầu vào: Sparse indices/values, shape và dense matrix.
# Đầu ra: Dense tensor là kết quả matrix multiplication.
# Lưu ý: Chỉ dùng khi membership weights cần học gradient.
class _SparseValuesMM(torch.autograd.Function):
    @staticmethod
    # Mục đích: Thực hiện forward sparse matrix nhân dense matrix.
    # Đầu vào: Autograd context, COO indices/values, shape và dense tensor.
    # Đầu ra: Dense tensor [sparse rows, dense columns].
    def forward(
        ctx: object,
        indices: torch.Tensor,
        values: torch.Tensor,
        rows: int,
        columns: int,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(indices, values, dense)
        ctx.shape = (rows, columns)
        matrix = torch.sparse_coo_tensor(
            indices,
            values,
            size=(rows, columns),
            dtype=values.dtype,
            device=values.device,
            check_invariants=False,
        ).coalesce()
        return torch.sparse.mm(matrix, dense)

    @staticmethod
    # Mục đích: Tính gradient cho sparse values và dense operand.
    # Đầu vào: Context đã lưu và gradient từ output.
    # Đầu ra: Gradient của values và dense; các shape/index input không có gradient.
    # Lưu ý: Chia chunk để không tạo tensor tạm quá lớn theo toàn bộ nnz.
    def backward(
        ctx: object, gradient: torch.Tensor
    ) -> tuple[None, torch.Tensor, None, None, torch.Tensor]:
        indices, values, dense = ctx.saved_tensors
        rows, columns = ctx.shape
        value_gradient = torch.empty_like(values)
        for start in range(0, values.numel(), SPARSE_BACKWARD_CHUNK):
            stop = min(start + SPARSE_BACKWARD_CHUNK, values.numel())
            row = indices[0, start:stop]
            column = indices[1, start:stop]
            value_gradient[start:stop] = (
                gradient[row] * dense[column]
            ).sum(dim=1)
        transpose = torch.sparse_coo_tensor(
            indices.flip(0),
            values.detach(),
            size=(columns, rows),
            dtype=values.dtype,
            device=values.device,
            check_invariants=False,
        ).coalesce()
        dense_gradient = torch.sparse.mm(transpose, gradient)
        return None, value_gradient, None, None, dense_gradient


# Mục đích: Chọn phép sparse matmul phù hợp với incidence cố định hoặc learnable.
# Đầu vào: PyTorch sparse matrix và dense tensor.
# Đầu ra: Dense multiplication result.
# Lưu ý: Dùng custom autograd khi sparse values có requires_grad=True.
def sparse_values_mm(matrix: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    matrix = matrix.coalesce()
    if matrix.values().requires_grad:
        return _SparseValuesMM.apply(
            matrix.indices(),
            matrix.values(),
            matrix.shape[0],
            matrix.shape[1],
            dense,
        )
    return torch.sparse.mm(matrix, dense)


# Mục đích: Chuyển SciPy sparse matrix sang PyTorch coalesced COO tensor.
# Đầu vào: SciPy sparse matrix và device đích.
# Đầu ra: PyTorch float32 sparse tensor cùng shape.
# Lưu ý: Coalesce gộp duplicate indices trước khi propagation.
def scipy_to_torch_sparse(
    matrix: sparse.spmatrix,
    *,
    device: torch.device,
) -> torch.Tensor:
    coo = matrix.tocoo(copy=False)
    indices = torch.from_numpy(
        np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        dtype=torch.float32,
        device=device,
        check_invariants=False,
    ).coalesce()


# Mục đích: Lưu H và các degree factors của công thức HGNN propagation.
# Đầu vào: Sparse incidence và các vector normalization đã tính.
# Đầu ra: Object bất biến có hàm propagate(node_features).
# Lưu ý: Không materialize ma trận truyền N x N.
@dataclass(frozen=True)
class HypergraphOperator:
    incidence: torch.Tensor
    incidence_t: torch.Tensor
    node_inverse_sqrt_degree: torch.Tensor
    edge_weight_over_degree: torch.Tensor

    @classmethod
    # Mục đích: Tạo operator differentiable từ weighted PyTorch incidence.
    # Đầu vào: Sparse incidence và edge weights tùy chọn.
    # Đầu ra: HypergraphOperator với degree normalization đã tính.
    # Lưu ý: Từ chối edge rỗng hoặc node cô lập để tránh chia cho 0.
    def from_torch_sparse(
        cls,
        incidence: torch.Tensor,
        *,
        edge_weights: torch.Tensor | None = None,
    ) -> "HypergraphOperator":
        if not incidence.is_sparse or incidence.ndim != 2:
            raise ValueError("incidence must be a sparse COO tensor")
        incidence = incidence.coalesce()
        node_count, edge_count = incidence.shape
        if node_count == 0 or edge_count == 0 or incidence._nnz() == 0:
            raise ValueError("incidence must be non-empty")
        indices = incidence.indices()
        values = incidence.values()
        if not torch.isfinite(values).all() or torch.any(values <= 0):
            raise ValueError("incidence values must be finite and positive")
        if edge_weights is None:
            weights = torch.ones(
                edge_count, dtype=values.dtype, device=values.device
            )
        else:
            weights = edge_weights.to(device=values.device, dtype=values.dtype)
        if weights.shape != (edge_count,) or torch.any(weights <= 0):
            raise ValueError("edge_weights must be positive and match edge count")
        edge_degree = torch.zeros(
            edge_count, dtype=values.dtype, device=values.device
        ).scatter_add(0, indices[1], values)
        node_degree = torch.zeros(
            node_count, dtype=values.dtype, device=values.device
        ).scatter_add(0, indices[0], values * weights[indices[1]])
        if torch.any(edge_degree <= 0) or torch.any(node_degree <= 0):
            raise ValueError("incidence cannot contain empty nodes or hyperedges")
        return cls(
            incidence=incidence,
            incidence_t=incidence.transpose(0, 1).coalesce(),
            node_inverse_sqrt_degree=node_degree.rsqrt(),
            edge_weight_over_degree=weights / edge_degree,
        )

    @classmethod
    # Mục đích: Tạo operator từ SciPy binary incidence H0.
    # Đầu vào: SciPy matrix, device và edge weights tùy chọn.
    # Đầu ra: HypergraphOperator float32 trên device yêu cầu.
    # Lưu ý: H0 phải binary; weighted H* dùng from_torch_sparse().
    def from_scipy(
        cls,
        incidence: sparse.spmatrix,
        *,
        device: torch.device,
        edge_weights: np.ndarray | None = None,
    ) -> "HypergraphOperator":
        matrix = incidence.tocsr().astype(np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("incidence must be a non-empty 2-D sparse matrix")
        if matrix.nnz == 0 or not np.all(matrix.data == 1):
            raise ValueError("incidence must be binary with at least one incidence")
        if edge_weights is None:
            weights = np.ones(matrix.shape[1], dtype=np.float32)
        else:
            weights = np.asarray(edge_weights, dtype=np.float32)
        if weights.shape != (matrix.shape[1],) or np.any(weights <= 0):
            raise ValueError("edge_weights must be positive and match edge count")

        edge_degree = np.asarray(matrix.sum(axis=0)).reshape(-1)
        node_degree = np.asarray(matrix @ weights).reshape(-1)
        if np.any(edge_degree <= 0) or np.any(node_degree <= 0):
            raise ValueError("incidence cannot contain empty nodes or hyperedges")
        torch_incidence = scipy_to_torch_sparse(matrix, device=device)
        return cls(
            incidence=torch_incidence,
            incidence_t=torch_incidence.transpose(0, 1).coalesce(),
            node_inverse_sqrt_degree=torch.as_tensor(
                np.power(node_degree, -0.5), dtype=torch.float32, device=device
            ),
            edge_weight_over_degree=torch.as_tensor(
                weights / edge_degree, dtype=torch.float32, device=device
            ),
        )

    # Mục đích: Áp dụng Dv^-1/2 H W De^-1 H^T Dv^-1/2 lên node features.
    # Đầu vào: Dense node feature hoặc embedding matrix [N,D].
    # Đầu ra: Dense propagated matrix [N,D].
    # Lưu ý: Hai phép sparse-dense tránh tạo adjacency N x N.
    def propagate(self, node_features: torch.Tensor) -> torch.Tensor:
        if node_features.ndim != 2:
            raise ValueError("node_features must be a 2-D tensor")
        if node_features.shape[0] != self.incidence.shape[0]:
            raise ValueError("Feature rows must match incidence rows")
        scaled_nodes = node_features * self.node_inverse_sqrt_degree[:, None]
        edge_messages = sparse_values_mm(self.incidence_t, scaled_nodes)
        edge_messages = edge_messages * self.edge_weight_over_degree[:, None]
        node_messages = sparse_values_mm(self.incidence, edge_messages)
        return node_messages * self.node_inverse_sqrt_degree[:, None]


# Mục đích: Một HGNN layer gồm linear projection rồi hypergraph propagation.
# Đầu vào: input_dim/output_dim khi tạo; features/operator khi forward.
# Đầu ra: Node representations [N, output_dim].
class HGNNLayer(nn.Module):
    # Mục đích: Khởi tạo linear projection và learnable bias.
    # Đầu vào: Số chiều input và output.
    # Đầu ra: HGNNLayer sẵn sàng train.
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    # Mục đích: Chiếu feature rồi truyền message qua hypergraph.
    # Đầu vào: Node features và HypergraphOperator.
    # Đầu ra: Node representations sau một layer.
    def forward(
        self, node_features: torch.Tensor, operator: HypergraphOperator
    ) -> torch.Tensor:
        return operator.propagate(self.linear(node_features)) + self.bias


# Mục đích: Baseline hai layer HGNN kèm binary dropout classifier.
# Đầu vào: input_dim, hidden_dim và dropout khi khởi tạo.
# Đầu ra: Encoder dùng độc lập hoặc logits qua forward().
class HGNNBaseline(nn.Module):
    # Mục đích: Khởi tạo hai HGNN layers, dropout và linear classifier.
    # Đầu vào: Số chiều feature, embedding và dropout probability.
    # Đầu ra: Model parameters có thể train bằng PyTorch.
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.layer1 = HGNNLayer(input_dim, hidden_dim)
        self.layer2 = HGNNLayer(hidden_dim, hidden_dim)
        # Dùng nn.Module trực tiếp để giữ checkpoint key classifier.output.*.
        # Cách này tránh tạo thêm một class chỉ chứa đúng một Linear layer.
        self.classifier = nn.Module()
        self.classifier.add_module("output", nn.Linear(hidden_dim, 1))
        self.dropout = dropout

    # Mục đích: Tạo node embedding từ feature và một hypergraph operator.
    # Đầu vào: Node features [N,F] và operator của H0 hoặc H*.
    # Đầu ra: Embedding [N, hidden_dim].
    # Lưu ý: ReLU/dropout nằm giữa hai HGNN layers.
    def encode(
        self, node_features: torch.Tensor, operator: HypergraphOperator
    ) -> torch.Tensor:
        hidden = torch.relu(self.layer1(node_features, operator))
        hidden = torch.nn.functional.dropout(
            hidden, p=self.dropout, training=self.training
        )
        return torch.relu(self.layer2(hidden, operator))

    # Mục đích: Chuyển node embedding thành dropout logit.
    # Đầu vào: Embeddings [N, hidden_dim].
    # Đầu ra: Logits [N], chưa áp dụng sigmoid.
    def classify(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier.output(embeddings).squeeze(-1)

    # Mục đích: Chạy baseline end-to-end trên một hypergraph.
    # Đầu vào: Node features và HypergraphOperator.
    # Đầu ra: Tuple (logits [N], embeddings [N,D]).
    def forward(
        self, node_features: torch.Tensor, operator: HypergraphOperator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(node_features, operator)
        return self.classify(embeddings), embeddings
