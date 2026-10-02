# 11. Baseline HyperGCN (Yadati et al., NeurIPS 2019) on the main model's hypergraph.
#
#   python src/11_hypergcn.py --mode both --seeds 1 --user-rule temporal
#
# Original code: baseline/HyperGCN (github.com/malllabiisc/HyperGCN, commit de04938).
# Its network (model/networks.py) and layer (model/utils.HyperGraphConvolution)
# are used unchanged. HyperGCN replaces each hyperedge by a small graph, again in
# every layer and epoch from the current features (non-fast HyperGCN): project the
# members on a random vector, link the two extremes Se and Ie, and with mediators
# link both to every other member, each edge with weight 1/(2|e| - 3).
#
# What differs from the original, and why:
#   - utils.Laplacian loops over hyperedges in Python; on the XuetangX graph (a few
#     million memberships, rebuilt 2 x 200 times) that takes hours, so it is
#     replaced by `laplacian` below, the same computation with array operations
#     (same random projection, first argmax/argmin, weights, + I, D^-1/2 A D^-1/2).
#     python src/11_hypergcn.py --check-laplacian compares the two on random graphs.
#   - Hypergraph = the bundle of the main model (4_hypergraph.py): Course, Object,
#     User hyperedges of train enrollments; validation/test targets are scored on
#     their local graphs with 8_train.predict, as the main model.
#   - --user-rule temporal: hypergraph_temporal.npz, and a node only receives the
#     graph edges of hyperedges whose members all started no later (the --causal
#     rule of 5_model.py). --user-rule any: hypergraph.npz, no causal mask.
#   - X is used as standardized by 3_features.py. The original row-normalizes
#     non-negative bag-of-words features; on standardized features the row sums
#     are near zero, so that normalization is skipped.
#   - Two classes with log-softmax and NLL, as the original; P(dropout) =
#     softmax class 1. The checkpoint is chosen every 10 epochs on validation AUPRC.
# Hyperparameters of the original config/config.py: depth 2 (hidden 16), dropout
# 0.5, lr 0.01, weight decay 5e-4, 200 epochs, mediators on.

import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

config = import_module("0_config")
data_module = import_module("10_baseline_data")
train_module = import_module("8_train")
hypergraph_module = import_module("4_hypergraph")
graph_to_device = import_module("5_model").graph_to_device
log = train_module.log

HYPERGCN_ROOT = data_module.BASELINE_ROOT / "HyperGCN"
SELF_LOOP = config.EDGE_FAMILIES.index("self_loop")
DEFAULT_SETTINGS = {
    "model": "hypergcn",
    "families": "course,object,user",
    "depth": 2,
    "dropout": 0.5,
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
    "epochs": 200,
    "mediators": 1,
    "eval_every": 10,
    "validation_limit": 5000,
    "eval_batch_size": 8,
    "feature_set": "full",
}
BUNDLES = {"temporal": "hypergraph_temporal.npz", "any": "hypergraph.npz"}


def import_hypergcn():
    if not HYPERGCN_ROOT.exists():
        raise FileNotFoundError(f"{HYPERGCN_ROOT} is missing; run bash scripts/setup_baselines.sh")
    sys.path.insert(0, str(HYPERGCN_ROOT))
    networks = import_module("model.networks")
    utils = import_module("model.utils")
    return networks, utils


# Graph approximation of a hypergraph (utils.Laplacian with array operations).
# structure: memberships of hyperedges with >= 2 members on one device, plus
# node_start and causal. X: features H·W of this layer (NumPy, as the original).
def laplacian(num_nodes, structure, X, mediators):
    node_ids, edge_ids = structure["node_ids"], structure["edge_ids"]
    device = node_ids.device
    edge_count = structure["num_edges"]
    rv = np.random.rand(X.shape[1])
    projection = torch.as_tensor(np.asarray(X, dtype=np.float64) @ rv, device=device)[node_ids]
    position = torch.arange(len(node_ids), device=device)

    def first_extreme(reduce):
        extreme = torch.full((edge_count,), -np.inf if reduce == "amax" else np.inf,
                             dtype=projection.dtype, device=device)
        extreme = extreme.scatter_reduce(0, edge_ids, projection, reduce=reduce)
        is_extreme = projection == extreme[edge_ids]
        first = torch.full((edge_count,), len(node_ids), dtype=torch.int64, device=device)
        first = first.scatter_reduce(0, edge_ids[is_extreme], position[is_extreme], reduce="amin")
        return first

    size = torch.bincount(edge_ids, minlength=edge_count)
    used = size >= 2
    supremum = torch.zeros(edge_count, dtype=torch.int64, device=device)
    infimum = torch.zeros(edge_count, dtype=torch.int64, device=device)
    supremum[used] = node_ids[first_extreme("amax")[used]]
    infimum[used] = node_ids[first_extreme("amin")[used]]
    used_edges = torch.nonzero(used).flatten()
    se, ie = supremum[used_edges], infimum[used_edges]

    if mediators:
        weight = 1.0 / (2.0 * size.double() - 3.0)
        rows = [se, ie]
        cols = [ie, se]
        edges = [used_edges, used_edges]
        mediator = (node_ids != supremum[edge_ids]) & (node_ids != infimum[edge_ids]) & used[edge_ids]
        m_nodes, m_edges = node_ids[mediator], edge_ids[mediator]
        rows += [supremum[m_edges], infimum[m_edges], m_nodes, m_nodes]
        cols += [m_nodes, m_nodes, supremum[m_edges], infimum[m_edges]]
        edges += [m_edges] * 4
    else:
        weight = 1.0 / size.double()
        rows, cols, edges = [se, ie], [ie, se], [used_edges, used_edges]
    rows, cols, edges = torch.cat(rows), torch.cat(cols), torch.cat(edges)
    values = weight[edges]

    if structure.get("causal"):
        start = structure["node_start"]
        latest = torch.full((edge_count,), torch.iinfo(torch.int64).min, dtype=torch.int64, device=device)
        latest = latest.scatter_reduce(0, edge_ids, start[node_ids], reduce="amax")
        receive = start[rows] >= latest[edges]
        rows, cols, values = rows[receive], cols[receive], values[receive]

    diagonal = torch.arange(num_nodes, device=device)
    rows = torch.cat([rows, diagonal])
    cols = torch.cat([cols, diagonal])
    values = torch.cat([values, torch.ones(num_nodes, dtype=values.dtype, device=device)])
    degree = torch.zeros(num_nodes, dtype=values.dtype, device=device).index_add_(0, rows, values)
    scale = degree.pow(-0.5)
    scale[torch.isinf(scale)] = 0
    values = scale[rows] * values * scale[cols]
    adjacency = torch.sparse_coo_tensor(torch.stack([rows, cols]), values.float(), (num_nodes, num_nodes),
                                        check_invariants=False)
    return adjacency.coalesce()


# Memberships HyperGCN uses: the graph without self-loops (the adjacency gets + I).
def hypergcn_structure(graph, causal):
    keep = graph["edge_family"][graph["edge_ids"]] != SELF_LOOP
    return {"node_ids": graph["node_ids"][keep], "edge_ids": graph["edge_ids"][keep],
            "num_edges": int(graph["num_edges"]), "node_start": graph["node_start"], "causal": causal}


class HyperGCNModel(nn.Module):
    def __init__(self, input_dim, settings, networks, use_cuda):
        super().__init__()
        args = SimpleNamespace(d=input_dim, c=2, depth=settings["depth"], cuda=use_cuda, fast=False,
                               mediators=bool(settings["mediators"]), dropout=settings["dropout"],
                               dataset="xuetangx")
        self.network = networks.HyperGCN(None, None, None, args)
        self.causal = settings["user_rule"] == "temporal"

    # 8_train.predict reads output["logits"]; sigmoid(z1 - z0) = softmax class 1.
    def forward(self, x, graph):
        self.network.structure = hypergcn_structure(graph, self.causal)
        log_probabilities = self.network(x)
        return {"logits": log_probabilities[:, 1] - log_probabilities[:, 0],
                "log_probabilities": log_probabilities}


def train_and_test(settings, seed, output_dir, device_name):
    data_module.set_seed(seed)
    device = data_module.resolve_device(device_name)
    networks, utils = import_hypergcn()
    utils.Laplacian = laplacian  # the layer calls utils.Laplacian(n, structure, HW, m)
    run_name = data_module.make_run_name(settings, seed)
    log("setup", f"run={run_name}, device={device}, settings={settings}")

    families = settings["families"].split(",")
    graph_settings = {"feature_set": settings["feature_set"], "families": families,
                      "hypergraph": BUNDLES[settings["user_rule"]], "shuffle_graph": None,
                      "eval_batch_size": settings["eval_batch_size"]}
    train_data = hypergraph_module.load_train_graph(
        output_dir, feature_set=settings["feature_set"], families=families,
        hypergraph_file=graph_settings["hypergraph"])
    validation_data = train_module.load_split(output_dir, "validation", graph_settings)
    if "user" in families and validation_data["user_rule"] != settings["user_rule"]:
        raise ValueError(f"{graph_settings['hypergraph']} has User rule {validation_data['user_rule']!r}, "
                         f"needs {settings['user_rule']!r}")
    x = torch.as_tensor(train_data["features"], device=device)
    labels = torch.as_tensor(train_data["labels"], device=device).long()
    graph = graph_to_device(train_data["graph"], device)
    limit = settings["limit"]
    train_rows = torch.arange(len(labels), device=device)
    if limit:
        train_rows = torch.as_tensor(np.random.default_rng(config.SPLIT_SEED).choice(
            len(labels), min(limit, len(labels)), replace=False), device=device)

    model = HyperGCNModel(x.shape[1], settings, networks, device.type == "cuda").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    checkpoint_path = Path(config.RUNS) / f"{run_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], {"score": -1.0, "epoch": 0}
    validation_limit = limit or settings["validation_limit"]
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        output = model(x, graph)
        loss = F.nll_loss(output["log_probabilities"][train_rows], labels[train_rows])
        loss.backward()
        optimizer.step()
        record = {"epoch": epoch, "loss": loss.item()}
        log("train", f"epoch {epoch}: loss={loss.item():.4f}")
        if epoch % settings["eval_every"] == 0 or epoch == settings["epochs"]:
            record["validation"] = data_module.epoch_validation(*train_module.predict(
                model, validation_data, graph_settings, device, limit=validation_limit,
                seed=config.SPLIT_SEED))
            if record["validation"]["auprc"] > best["score"]:
                best = {"score": record["validation"]["auprc"], "epoch": epoch}
                torch.save({"state_dict": model.state_dict(), "settings": settings, "seed": seed,
                            "epoch": epoch}, checkpoint_path)
                log("checkpoint", f"new best val_auprc={best['score']:.4f} at epoch {epoch}")
        history.append(record)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
    full_limit = limit  # 0 = the full split
    validation = train_module.predict(model, validation_data, graph_settings, device, limit=full_limit,
                                      seed=config.SPLIT_SEED)
    test_data = train_module.load_split(output_dir, "test", graph_settings)
    test = train_module.predict(model, test_data, graph_settings, device, limit=full_limit, seed=seed)
    data_module.write_reports(run_name, settings, seed, history=history, best_epoch=best["epoch"],
                              checkpoint_path=checkpoint_path, validation=validation, test=test)


# The array version against the original loop on small random hypergraphs.
def check_laplacian(trials=20):
    _, utils = import_hypergcn()
    original = utils.Laplacian
    rng = np.random.default_rng(0)
    for trial in range(trials):
        n = int(rng.integers(5, 40))
        hyperedges = [sorted(rng.choice(n, int(rng.integers(2, min(n, 8))), replace=False).tolist())
                      for _ in range(int(rng.integers(1, 12)))]
        x = rng.normal(size=(n, 4)).astype(np.float32)
        if trial % 4 == 0:
            x[:] = 1.0  # ties: Se = Ie
        for mediators in (True, False):
            np.random.seed(trial)
            expected = original(n, {k: e for k, e in enumerate(hyperedges)}, x, mediators).to_dense()
            structure = {"node_ids": torch.as_tensor(np.concatenate(hyperedges)),
                         "edge_ids": torch.as_tensor(np.repeat(np.arange(len(hyperedges)),
                                                               [len(e) for e in hyperedges])),
                         "num_edges": len(hyperedges), "node_start": torch.zeros(n, dtype=torch.int64),
                         "causal": False}
            np.random.seed(trial)
            found = laplacian(n, structure, x, mediators).to_dense()
            if not torch.allclose(expected, found, atol=1e-6):
                raise AssertionError(f"trial {trial}, mediators={mediators}: max diff "
                                     f"{(expected - found).abs().max():.2e}")
    log("check", f"laplacian matches utils.Laplacian on {trials} random hypergraphs (with/without mediators)")


if __name__ == "__main__":
    parser = data_module.base_parser("Train and test HyperGCN on the main hypergraph.", DEFAULT_SETTINGS)
    parser.add_argument("--check-laplacian", action="store_true")
    arguments = parser.parse_args()
    if arguments.check_laplacian:
        check_laplacian()
    else:
        settings = data_module.settings_from(arguments, DEFAULT_SETTINGS)
        for seed in arguments.seeds:
            train_and_test(settings, seed, arguments.output_dir, arguments.device)
