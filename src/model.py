"""Build sparse H0, align X, and run the two HGNN passes of one HGSL model."""

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from preprocess import PROCESSED, SEEDS, read_csv
from hsl import refine_hypergraph


def feature_columns(feature_set):
    if feature_set == "behavior":
        return slice(0, 60)
    if feature_set == "behavior_user":
        return slice(0, 75)
    if feature_set == "behavior_course":
        return np.r_[0:60, 75:96]
    if feature_set == "full":
        return slice(0, 96)
    raise ValueError(f"Unknown feature set: {feature_set}")


def build_h0(output_dir=PROCESSED, *, seed=1):
    """Turn train edge membership rows into a CSR H0 matrix."""
    output_dir = Path(output_dir)
    split = [row["split"] for row in read_csv(output_dir / f"split_seed_{seed}.csv")]
    train_ids = np.flatnonzero(np.array(split) == "train")
    global_to_train = {int(node): row for row, node in enumerate(train_ids)}
    meta = list(read_csv(output_dir / f"edge_meta_seed_{seed}.csv"))
    rows, cols = [], []
    with gzip.open(output_dir / f"edge_memberships_seed_{seed}.csv.gz", "rt",
                   newline="", encoding="utf-8") as source:
        for item in csv.DictReader(source):
            rows.append(global_to_train[int(item["node_id"])])
            cols.append(int(item["edge_id"]))
    h0 = sparse.coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                           shape=(len(train_ids), len(meta))).tocsr()
    if np.any(np.asarray(h0.sum(axis=0)).ravel() < 2):
        raise ValueError("H0 contains an empty or singleton edge")
    if np.any(np.asarray(h0.sum(axis=1)).ravel() == 0):
        raise ValueError("H0 contains an isolated train node")
    sparse.save_npz(output_dir / f"H0_seed_{seed}.npz", h0)
    np.save(output_dir / f"train_ids_seed_{seed}.npy", train_ids)
    report = {"seed": seed, "shape": h0.shape, "incidences": h0.nnz}
    print(report, flush=True)
    return report


def load_train_graph(output_dir=PROCESSED, *, seed=1, feature_set="behavior"):
    output_dir = Path(output_dir)
    h0 = sparse.load_npz(output_dir / f"H0_seed_{seed}.npz")
    train_ids = np.load(output_dir / f"train_ids_seed_{seed}.npy")
    x = np.load(output_dir / f"X_seed_{seed}.npy", mmap_mode="r")
    x = np.asarray(x[train_ids][:, feature_columns(feature_set)], dtype=np.float32)
    nodes = list(read_csv(output_dir / "nodes.csv"))
    labels = np.array([int(nodes[node]["label"]) for node in train_ids], dtype=np.float32)
    meta = list(read_csv(output_dir / f"edge_meta_seed_{seed}.csv"))
    families = np.array([row["family"] for row in meta])
    sizes = np.array([int(row["size"]) for row in meta], dtype=np.int64)
    return x, h0, labels, families, sizes


def load_evaluation_data(output_dir=PROCESSED, *, seed=1, feature_set="behavior"):
    """Read small node tables once; local validation/test edges use train references."""
    output_dir = Path(output_dir)
    nodes = list(read_csv(output_dir / "nodes.csv"))
    split = [row["split"] for row in read_csv(output_dir / f"split_seed_{seed}.csv")]
    course_refs, object_refs, objects_by_node = defaultdict(list), defaultdict(list), defaultdict(set)
    for node_id, node in enumerate(nodes):
        if split[node_id] == "train":
            course_refs[node["course_id"]].append(node_id)
    with gzip.open(output_dir / "node_objects.csv.gz", "rt", newline="",
                   encoding="utf-8") as source:
        for row in csv.DictReader(source):
            node_id = int(row["node_id"])
            key = (row["course_id"], row["object_id"], row["object_type"])
            objects_by_node[node_id].add(key)
            if split[node_id] == "train":
                object_refs[key].append(node_id)
    return {"nodes": nodes, "split": split, "course_refs": course_refs,
            "object_refs": object_refs, "objects_by_node": objects_by_node,
            "neighbors": np.load(output_dir / f"neighbors_seed_{seed}.npy"),
            "x": np.load(output_dir / f"X_seed_{seed}.npy"),
            "columns": feature_columns(feature_set)}


def local_graph(data, target, k=10):
    """One target plus only train references; return X, H0, edge families and label."""
    if data["split"][target] == "train":
        raise ValueError("A local evaluation target must be validation or test")
    course = data["nodes"][target]["course_id"]
    edges = []
    if data["course_refs"][course]:
        edges.append(("course", data["course_refs"][course]))
    for key in sorted(data["objects_by_node"][target]):
        refs = data["object_refs"][key]
        if refs:
            edges.append(("object", refs))
    edges.append(("behavioral", data["neighbors"][target, :k]))
    references = sorted({int(node) for _, refs in edges for node in refs})
    ids = np.array([target, *references], dtype=np.int64)
    positions = {node: i + 1 for i, node in enumerate(references)}
    rows, columns = [], []
    for edge_id, (_, refs) in enumerate(edges):
        rows.append(0)
        columns.append(edge_id)
        for node in refs:
            rows.append(positions[int(node)])
            columns.append(edge_id)
    h0 = sparse.coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, columns)),
                           shape=(len(ids), len(edges))).tocsr()
    x = np.asarray(data["x"][ids][:, data["columns"]], dtype=np.float32)
    families = np.array([family for family, _ in edges])
    sizes = np.asarray(h0.sum(axis=0)).ravel().astype(np.int64)
    label = int(data["nodes"][target]["label"])
    return x, h0, families, sizes, label


class _SparseValuesMM(torch.autograd.Function):
    """Backpropagate through stored sparse values without a dense N x E gradient."""

    @staticmethod
    def forward(ctx, indices, values, shape, dense):
        ctx.save_for_backward(indices, values, dense)
        ctx.shape = shape
        matrix = torch.sparse_coo_tensor(indices, values, shape,
                                         check_invariants=False).coalesce()
        return torch.sparse.mm(matrix, dense)

    @staticmethod
    def backward(ctx, gradient):
        indices, values, dense = ctx.saved_tensors
        value_gradient = torch.empty_like(values)
        for start in range(0, len(values), 100000):
            stop = min(start + 100000, len(values))
            row, col = indices[:, start:stop]
            value_gradient[start:stop] = (gradient[row] * dense[col]).sum(dim=1)
        transpose = torch.sparse_coo_tensor(indices.flip(0), values.detach(),
            (ctx.shape[1], ctx.shape[0]), check_invariants=False).coalesce()
        return None, value_gradient, None, torch.sparse.mm(transpose, gradient)


def _sparse_mm(matrix, dense):
    matrix = matrix.coalesce()
    if matrix.values().requires_grad:
        return _SparseValuesMM.apply(matrix.indices(), matrix.values(), matrix.shape, dense)
    return torch.sparse.mm(matrix, dense)


def to_torch_sparse(matrix, device):
    coo = matrix.tocoo(copy=False)
    indices = torch.as_tensor(np.stack((coo.row, coo.col)), dtype=torch.int64, device=device)
    values = torch.as_tensor(coo.data.astype(np.float32), device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape,
                                   check_invariants=False).coalesce()


def _operator(h):
    h = h.coalesce()
    row, col = h.indices()
    value = h.values()
    node_degree = torch.zeros(h.shape[0], device=value.device).scatter_add(0, row, value)
    edge_degree = torch.zeros(h.shape[1], device=value.device).scatter_add(0, col, value)
    if torch.any(node_degree <= 0) or torch.any(edge_degree <= 0):
        raise ValueError("H has an isolated node or empty edge")
    return h, h.transpose(0, 1).coalesce(), node_degree.rsqrt(), edge_degree.reciprocal()


def _propagate(operator, x):
    h, h_transpose, node_scale, edge_scale = operator
    edge_messages = _sparse_mm(h_transpose, x * node_scale[:, None])
    return _sparse_mm(h, edge_messages * edge_scale[:, None]) * node_scale[:, None]


class HGSLModel(nn.Module):
    """One shared two-layer HGNN, one membership scorer, and one classifier."""

    def __init__(self, input_dim, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.layer2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias2 = nn.Parameter(torch.zeros(hidden_dim))
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.membership_bias = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def encode(self, x, h):
        operator = _operator(h)
        hidden = F.relu(_propagate(operator, self.layer1(x)) + self.bias1)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return F.relu(_propagate(operator, self.layer2(hidden)) + self.bias2)

    def forward(self, x, h0, families, sizes, rng, *, hsl=True, deterministic=False,
                refinement=None, node_groups=None, edge_groups=None):
        refinement = refinement or {}
        initial = to_torch_sparse(h0, x.device)
        z0 = self.encode(x, initial)
        if hsl:
            h_star, audit = refine_hypergraph(
                z0, h0, families, sizes, self.node_projection, self.edge_projection,
                self.membership_bias, rng, deterministic=deterministic,
                node_groups=node_groups, edge_groups=edge_groups, **refinement)
            z_star = self.encode(x, h_star)
        else:
            h_star, z_star, audit = initial, z0, {"sampled_hyperedges": 0}
        logits = self.classifier(z_star).squeeze(-1)
        return {"logits": logits, "z0": z0, "z_star": z_star,
                "h_star": h_star, "audit": audit}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=1)
    args = parser.parse_args()
    build_h0(args.output_dir, seed=args.seed)
