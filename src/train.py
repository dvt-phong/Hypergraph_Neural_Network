"""Train HGSL, select a checkpoint on validation, then evaluate on test."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from scipy import sparse
import torch

from losses import total_loss, train_pos_weight
from model import HGSLModel, load_evaluation_data, load_train_graph, local_graph
from preprocess import PROCESSED, ROOT, SEEDS


RUNS = ROOT / "outputs" / "runs"
REPORTS = ROOT / "outputs" / "reports"


def artifact_hashes(output_dir, seed):
    """Identify the exact files used to train and evaluate one seed."""
    names = ("nodes.csv", "node_objects.csv.gz", f"split_seed_{seed}.csv",
             f"X_seed_{seed}.npy", f"neighbors_seed_{seed}.npy",
             f"H0_seed_{seed}.npz", f"train_ids_seed_{seed}.npy",
             f"edge_meta_seed_{seed}.csv", f"graph_config_seed_{seed}.json")
    hashes = {}
    for name in names:
        digest = hashlib.sha256()
        with open(Path(output_dir) / name, "rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return hashes


def metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or len(np.unique(labels)) != 2:
        raise ValueError("AUC needs both label classes and equal-length predictions")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    stops = np.r_[starts[1:], len(scores)]
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = (start + 1 + stop) / 2
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    descending = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[descending]
    ordered_scores = scores[descending]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0), len(scores) - 1]
    tp = np.cumsum(ordered_labels)[ends]
    precision_curve = tp / (ends + 1)
    recall_curve = tp / n_pos
    auprc = np.sum(np.diff(np.r_[0.0, recall_curve]) * precision_curve)
    predicted = scores >= 0.5
    true_positive = int(np.count_nonzero(predicted & (labels == 1)))
    false_positive = int(np.count_nonzero(predicted & (labels == 0)))
    false_negative = int(np.count_nonzero(~predicted & (labels == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"auc": float(auc), "auprc": float(auprc), "f1": float(f1),
            "precision": float(precision), "recall": float(recall)}


def _batch_graphs(graphs):
    features = np.concatenate([graph[0] for graph in graphs])
    incidence = sparse.block_diag([graph[1] for graph in graphs], format="csr")
    families = np.concatenate([graph[2] for graph in graphs])
    sizes = np.concatenate([graph[3] for graph in graphs])
    target_positions = np.cumsum([0] + [len(graph[0]) for graph in graphs[:-1]])
    node_groups = np.concatenate([np.full(len(graph[0]), i) for i, graph in enumerate(graphs)])
    edge_groups = np.concatenate([np.full(graph[1].shape[1], i)
                                  for i, graph in enumerate(graphs)])
    labels = np.array([graph[4] for graph in graphs], dtype=np.int8)
    return features, incidence, families, sizes, target_positions, node_groups, edge_groups, labels


def evaluate(model, data, split, *, seed, device, batch_size=4, limit=0,
             hsl=True, refinement=None, k=10):
    targets = np.array([i for i, name in enumerate(data["split"]) if name == split])
    if limit:
        if limit < 2:
            raise ValueError("Evaluation limit must be zero or at least two")
        rng = np.random.default_rng(seed)
        negative = [i for i in targets if data["nodes"][i]["label"] == "0"]
        positive = [i for i in targets if data["nodes"][i]["label"] == "1"]
        selected = [int(rng.choice(negative)), int(rng.choice(positive))]
        remaining = np.setdiff1d(targets, selected)
        extra = rng.choice(remaining, min(limit - 2, len(remaining)), replace=False)
        targets = np.sort(np.r_[selected, extra])
    probabilities, labels = [], []
    audits = {"sampled_hyperedges": 0, "membership_candidates": 0,
              "restored_isolated_nodes": 0}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(targets), batch_size):
            graphs = [local_graph(data, int(target), k) for target in targets[start:start + batch_size]]
            x, h0, families, sizes, positions, node_groups, edge_groups, batch_labels = (
                _batch_graphs(graphs))
            output = model(torch.as_tensor(x, device=device), h0, families, sizes,
                           np.random.default_rng(seed), hsl=hsl, deterministic=True,
                           refinement=refinement, node_groups=node_groups,
                           edge_groups=edge_groups)
            probabilities.extend(torch.sigmoid(output["logits"][positions]).cpu().tolist())
            labels.extend(batch_labels.tolist())
            for key in audits:
                audits[key] += int(output["audit"].get(key, 0))
    return metrics(labels, probabilities), audits, len(targets)


def train(*, seed=1, feature_set="behavior", epochs=50, patience=5,
          output_dir=PROCESSED, device_name="auto", validation_limit=0,
          validation_batch_size=4, hidden_dim=64, dropout=0.5,
          learning_rate=0.001, weight_decay=0.0005, lambda_cl=0.1,
          temperature=0.2, contrastive_nodes=512, k=None, hsl=True,
          refinement=None, runs_dir=RUNS, reports_dir=REPORTS):
    """One full-batch training run; labels from validation choose the checkpoint."""
    output_dir = Path(output_dir)
    graph_k = json.loads((output_dir / f"graph_config_seed_{seed}.json").read_text(
        encoding="utf-8"))["k"]
    if k is None:
        k = graph_k
    elif k != graph_k:
        raise ValueError(f"Train k={k} differs from graph k={graph_k}")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, h0, y, families, sizes = load_train_graph(output_dir, seed=seed,
                                                  feature_set=feature_set)
    data = load_evaluation_data(output_dir, seed=seed, feature_set=feature_set)
    x_tensor = torch.as_tensor(x, device=device)
    labels = torch.as_tensor(y, device=device)
    pos_weight = train_pos_weight(labels)
    model = HGSLModel(x.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    refinement = {"sampled_hyperedges": 96, "positive_nodes": 16,
                  "negative_nodes": 16, "top_r": 8, "threshold": None,
                  **(refinement or {})}
    settings = {"seed": seed, "feature_set": feature_set, "epochs": epochs,
                "patience": patience, "validation_limit": validation_limit,
                "validation_batch_size": validation_batch_size,
                "hidden_dim": hidden_dim, "dropout": dropout,
                "learning_rate": learning_rate, "weight_decay": weight_decay,
                "lambda_cl": lambda_cl, "temperature": temperature,
                "contrastive_nodes": contrastive_nodes, "k": k, "hsl": hsl,
                "refinement": refinement}
    artifacts = artifact_hashes(output_dir, seed)
    run_id = hashlib.sha256(json.dumps({"settings": settings, "artifacts": artifacts},
        sort_keys=True).encode("utf-8")).hexdigest()[:10]
    runs_dir, reports_dir = Path(runs_dir), Path(reports_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    name = f"{'hgsl' if hsl else 'hgnn'}_{feature_set}_seed_{seed}"
    checkpoint = runs_dir / f"simple_{name}_{run_id}.pt"
    history, best_auc, best_epoch, stale = [], -float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(seed * 100003 + epoch)
        sampled = rng.choice(len(y), min(contrastive_nodes, len(y)), replace=False)
        sampled = torch.as_tensor(sampled, dtype=torch.int64, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(x_tensor, h0, families, sizes, rng, hsl=hsl,
                       refinement=refinement)
        loss, parts = total_loss(output, labels, pos_weight, sampled,
                                 lambda_cl=lambda_cl if hsl else 0,
                                 temperature=temperature)
        if not torch.isfinite(loss):
            raise ValueError("Training loss is not finite")
        loss.backward()
        scorer_norm = float(torch.sqrt(sum((parameter.grad.square().sum()
            for parameter in (model.node_projection.weight, model.edge_projection.weight,
                              model.membership_bias) if parameter.grad is not None),
            torch.zeros((), device=device))).detach())
        if hsl and scorer_norm <= 0:
            raise ValueError("No gradient reached the HSL membership scorer")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        validation, audit, count = evaluate(model, data, "validation", seed=seed,
            device=device, batch_size=validation_batch_size, limit=validation_limit,
            hsl=hsl, refinement=refinement, k=k)
        history.append({"epoch": epoch, "loss": float(loss.detach()), **parts,
                        "scorer_gradient_norm": scorer_norm, "validation": validation,
                        "train_refinement": output["audit"], "validation_refinement": audit})
        print(f"epoch {epoch}: loss={float(loss.detach()):.4f} val_auc={validation['auc']:.4f}",
              flush=True)
        if validation["auc"] > best_auc:
            best_auc, best_epoch, stale = validation["auc"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "validation": validation, "seed": seed, "feature_set": feature_set,
                        "input_dim": x.shape[1], "hidden_dim": hidden_dim,
                        "dropout": dropout, "hsl": hsl, "refinement": refinement,
                        "k": k, "settings": settings,
                        "artifact_hashes": artifacts}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    report = {"checkpoint": str(checkpoint), "best_epoch": best_epoch,
              "best_validation": history[best_epoch - 1]["validation"],
              "validation_targets": count, "history": history,
              "seed": seed, "feature_set": feature_set, "train_nodes": len(y),
              "test_used_for_selection": False, "pos_weight": float(pos_weight),
              "settings": settings, "artifact_hashes": artifacts}
    (reports_dir / f"simple_{name}_{run_id}_train.json").write_text(json.dumps(report, indent=2),
                                                              encoding="utf-8")
    print(f"Saved validation-selected checkpoint: {checkpoint}", flush=True)
    return report


def test(checkpoint, *, output_dir=PROCESSED, device_name="auto", test_limit=0,
         batch_size=4, reports_dir=REPORTS):
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if "artifact_hashes" not in saved:
        raise ValueError("Checkpoint has no data hashes; retrain with the current code")
    current_hashes = artifact_hashes(output_dir, saved["seed"])
    changed = [name for name, digest in saved["artifact_hashes"].items()
               if current_hashes.get(name) != digest]
    if changed:
        raise ValueError(f"Processed data changed since training: {', '.join(changed)}")
    model = HGSLModel(saved["input_dim"], saved["hidden_dim"], saved["dropout"]).to(device)
    model.load_state_dict(saved["state_dict"])
    data = load_evaluation_data(output_dir, seed=saved["seed"],
                                feature_set=saved["feature_set"])
    results, audit, count = evaluate(model, data, "test", seed=saved["seed"],
        device=device, batch_size=batch_size, limit=test_limit, hsl=saved["hsl"],
        refinement=saved["refinement"], k=saved["k"])
    report = {"checkpoint": str(checkpoint), "checkpoint_epoch": saved["epoch"],
              "checkpoint_validation": saved["validation"], "test_targets": count,
              "test_metrics": results, "test_refinement": audit}
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / (Path(checkpoint).stem + "_test.json")).write_text(json.dumps(report,
        indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "test", "both"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=1)
    parser.add_argument("--feature-set", choices=("behavior", "behavior_user",
                        "behavior_course", "full"), default="behavior")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-hsl", action="store_true")
    args = parser.parse_args()
    if args.mode in ("train", "both"):
        result = train(seed=args.seed, feature_set=args.feature_set, epochs=args.epochs,
                       output_dir=args.output_dir, device_name=args.device,
                       validation_limit=args.validation_limit, hsl=not args.no_hsl)
        args.checkpoint = Path(result["checkpoint"])
    if args.mode in ("test", "both"):
        if args.checkpoint is None:
            parser.error("--checkpoint is required for --mode test")
        print(test(args.checkpoint, output_dir=args.output_dir,
                   device_name=args.device, test_limit=args.test_limit), flush=True)
