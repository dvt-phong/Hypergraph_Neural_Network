# 9. Run the experiment scenarios (config.SCENARIOS, docs/KICH_BAN_THUC_NGHIEM.md).
#    For every scenario and every seed: train on the train hypergraph H0 with early
#    stopping on validation AUC, score validation and test once with the best weights at
#    threshold 0.5, and append one row to outputs/results.csv. No other file is saved.
#
#   python src/9_train.py --scenario all --seeds 1 11 111 1111 11111     all 11 scenarios × 5 seeds
#   python src/9_train.py --scenario M A4 B1                             some scenarios
#   python src/9_train.py --scenario all --seeds 1 --epochs 10 --eval-limit 2000   quick check
# Tham khảo từ project/bài báo:
# - HGNN, AAAI 2019 (Feng et al.): hidden size, dropout, optimizer settings
#   Code: https://github.com/iMoonLab/HGNN

import argparse
import copy
import csv
import random
import time
from importlib import import_module

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.nn import functional as F

config = import_module("0_config")
graph_data = import_module("5_graph_data")
prepare_graph = import_module("6_hgnn").prepare_graph
DropoutModel = import_module("8_model").DropoutModel

RESULT_COLUMNS = (
    "time", "scenario", "seed",
    "families", "features", "hgnn_layers", "use_mlp", "learn_w",
    "epochs", "eval_every", "patience", "hidden_dim", "dropout", "learning_rate", "weight_decay",
    "best_epoch", "epochs_run",
    "val_auc", "val_auprc", "val_f1",
    "test_auc", "test_auprc", "test_accuracy", "test_precision", "test_recall", "test_f1",
    "w_course", "w_object", "w_user", "w_self_loop",
    "minutes",
)


# Print a message with the time and a scope, e.g. [train].
def log(scope, message):
    print(f"[{time.strftime('%H:%M:%S')}][{scope}] {message}", flush=True)


# Seed Python, NumPy and PyTorch.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Input:  "auto", "cpu" or "cuda".
# Output: torch.device ("auto" = cuda when available).
def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# Metrics of the dropout class (label 1) at threshold 0.5.
# Input:  labels [n] (0/1), probabilities [n].
# Output: dict auc, auprc, accuracy, precision, recall, f1.
def metrics(labels, probabilities):
    predicted = probabilities >= config.THRESHOLD                       # ŷ = 1[p ≥ 0.5]
    precision, recall, f1, _ = precision_recall_fscore_support(         # P = TP/(TP+FP), R = TP/(TP+FN),
        labels, predicted, labels=[1], zero_division=0)                  # F1 = 2PR/(P+R)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),             # AUC = P(p_dropout > p_non-dropout)
        "auprc": float(average_precision_score(labels, probabilities)), # AUPRC = Σ_n (R_n − R_{n−1})·P_n
        "accuracy": float(np.mean(predicted == labels)),                # Acc = (TP + TN)/n
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
    }


# Dropout probabilities of validation/test targets (steps A and B of 6_hgnn.py).
# Input:  model, train features x and graph (tensors), targets from apply_scenario,
#         device, batch size.
# Output: probabilities [T] (NumPy).
@torch.no_grad()
def predict(model, x, graph, targets, device, batch_size):
    was_training = model.training
    model.eval()                                                         # no dropout
    cache = model.cache_train_states(x, graph)                           # step A, once
    ptr = targets["h0_ptr"]
    probabilities = []
    for start in range(0, len(targets["labels"]), batch_size):
        stop = min(start + batch_size, len(targets["labels"]))
        counts = np.diff(ptr[start:stop + 1])                            # |E(t) ∩ H0| per target
        rows = torch.as_tensor(np.repeat(np.arange(stop - start), counts), device=device)
        edges = torch.as_tensor(targets["h0_edges"][ptr[start]:ptr[stop]], device=device)
        single_user = torch.as_tensor(targets["single_user"][start:stop], device=device)
        features = torch.as_tensor(targets["features"][start:stop], device=device)
        logits = model.forward_targets(features, rows, edges, single_user, cache)   # step B
        probabilities.append(torch.sigmoid(logits).cpu().numpy())        # p = σ(logit)
    model.train(was_training)
    return np.concatenate(probabilities)


# Append one row to outputs/results.csv (header written when the file is new).
def append_result(row):
    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    is_new = not config.RESULTS_CSV.exists()
    with open(config.RESULTS_CSV, "a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({name: f"{value:.6f}" if isinstance(value, float) else value
                         for name, value in row.items()})
    log("result", f"appended to {config.RESULTS_CSV}")


# Train one scenario with one seed, early stopping on validation AUC, then score
# validation and test with the best weights.
# Input:  scenario code (e.g. "A4"), seed, settings (config.TRAIN with overrides),
#         data of the scenario (from apply_scenario), device.
# Output: none (one row appended to results.csv).
def train_one_seed(code, seed, settings, data, device):
    started_at = time.perf_counter()
    scenario = config.SCENARIOS[code]
    set_seed(seed)
    x = torch.as_tensor(data["train"]["features"], device=device)
    labels = torch.as_tensor(data["train"]["labels"], device=device)
    graph = prepare_graph(data["train"]["graph"], device)
    validation_labels = data["validation"]["labels"]

    model = DropoutModel(x.shape[1], settings["hidden_dim"], settings["dropout"],
                         hgnn_layers=scenario["hgnn_layers"], use_mlp=scenario["use_mlp"],
                         learn_w=scenario["learn_w"]).to(device)
    family_logits = model.hgnn.family_logits
    groups = [{"params": [p for p in model.parameters() if p is not family_logits]}]   # Θ, A, u: lr, L2
    if scenario["learn_w"]:                                                          # W: own lr, no L2
        groups.append({"params": [family_logits], "lr": settings["family_weight_lr"], "weight_decay": 0.0})
    optimizer = torch.optim.Adam(groups, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])

    best = {"auc": -1.0, "epoch": 0, "state": None}
    stale = 0          # validations in a row without a better AUC
    epoch = 0
    for epoch in range(1, settings["epochs"] + 1):
        # One full-batch training step on H0.
        model.train()
        optimizer.zero_grad()
        logits = model(x, graph)
        loss = F.binary_cross_entropy_with_logits(logits, labels)       # L = −(1/N)·Σ [y·log p + (1−y)·log(1−p)]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)          # ‖g‖ ≤ 5
        optimizer.step()

        if epoch % 50 == 0:
            train_auc = roc_auc_score(data["train"]["labels"], logits.detach().cpu().numpy())
            log("train", f"{code} seed {seed} epoch {epoch}: loss={loss.item():.4f}, train_auc={train_auc:.4f}")

        # Early stopping: validation AUC every eval_every epochs, keep the best weights in memory.
        if epoch % settings["eval_every"] == 0:
            validation_auc = roc_auc_score(validation_labels, predict(
                model, x, graph, data["validation"], device, settings["eval_batch_size"]))
            if validation_auc > best["auc"]:                             # best = argmax_epoch AUC_val
                best = {"auc": validation_auc, "epoch": epoch,
                        "state": copy.deepcopy(model.state_dict())}      # kept in memory, no file
                stale = 0
            else:
                stale += 1
            if epoch % 50 == 0 or stale == 0:
                log("validation", f"{code} seed {seed} epoch {epoch}: val_auc={validation_auc:.4f} "
                                  f"(best {best['auc']:.4f} at epoch {best['epoch']})")
            if stale >= settings["patience"]:
                log("train", f"{code} seed {seed}: early stopping at epoch {epoch}")
                break

    # Score validation and test once, with the best weights.
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    scores = {}
    for split in ("validation", "test"):
        probabilities = predict(model, x, graph, data[split], device, settings["eval_batch_size"])
        scores[split] = metrics(data[split]["labels"], probabilities)
        log(split, f"{code} seed {seed}: " + ", ".join(f"{name}={value:.4f}"
                                                       for name, value in scores[split].items()))

    append_result({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "scenario": code, "seed": seed,
        "families": "+".join(scenario["families"]), "features": scenario["features"],
        "hgnn_layers": scenario["hgnn_layers"], "use_mlp": scenario["use_mlp"],
        "learn_w": scenario["learn_w"],
        **{name: settings[name] for name in ("epochs", "eval_every", "patience", "hidden_dim",
                                             "dropout", "learning_rate", "weight_decay")},
        "best_epoch": best["epoch"], "epochs_run": epoch,
        **{f"val_{name}": scores["validation"][name] for name in ("auc", "auprc", "f1")},
        **{f"test_{name}": value for name, value in scores["test"].items()},
        **model.weight_summary(scenario["families"]),
        "minutes": (time.perf_counter() - started_at) / 60,
    })


# --scenario all -> config.SCENARIOS_ALL; otherwise the given codes, checked.
def scenario_codes(values):
    codes = list(config.SCENARIOS_ALL) if values == ["all"] else values
    unknown = [code for code in codes if code not in config.SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenarios {unknown}; choose from {list(config.SCENARIOS)} or 'all'")
    return codes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment scenarios: train with early stopping "
                                                 "on validation AUC, then score validation and test at 0.5.")
    parser.add_argument("--scenario", nargs="+", default=["M"],
                        help=f"'all' ({' '.join(config.SCENARIOS_ALL)}) or codes from {list(config.SCENARIOS)}")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    parser.add_argument("--epochs", type=int, default=config.TRAIN["epochs"])
    parser.add_argument("--eval-every", type=int, default=config.TRAIN["eval_every"])
    parser.add_argument("--patience", type=int, default=config.TRAIN["patience"])
    parser.add_argument("--eval-limit", type=int, default=0,
                        help="score only the first N validation/test targets (quick check; 0 = all)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default=config.PROCESSED)
    arguments = parser.parse_args()

    codes = scenario_codes(arguments.scenario)
    settings = {**config.TRAIN, "epochs": arguments.epochs,
                "eval_every": arguments.eval_every, "patience": arguments.patience}
    device = resolve_device(arguments.device)
    log("setup", f"device={device}, scenarios={codes}, seeds={arguments.seeds}, settings={settings}")

    # Read the data once (all families, all columns); every scenario takes its part of it.
    loaded = {"train": graph_data.load_train_graph(arguments.output_dir)}
    for split in ("validation", "test"):
        loaded[split] = graph_data.load_targets(arguments.output_dir, split_name=split,
                                                limit=arguments.eval_limit)
        log("setup", f"{split}: {len(loaded[split]['labels']):,} targets")

    # Scenario by scenario; every seed of a scenario before the next one.
    for code in codes:
        scenario = config.SCENARIOS[code]   # the Vietnamese description is not logged (Windows console)
        log("scenario", f"{code}: families={'+'.join(scenario['families'])}, features={scenario['features']}, "
                        f"hgnn_layers={scenario['hgnn_layers']}, use_mlp={scenario['use_mlp']}, "
                        f"learn_w={scenario['learn_w']}")
        data = graph_data.apply_scenario(loaded, config.SCENARIOS[code])
        for seed in arguments.seeds:
            train_one_seed(code, seed, settings, data, device)
