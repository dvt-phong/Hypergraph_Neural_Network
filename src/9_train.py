# 9. Run the experiment scenarios 
# - HGNN https://github.com/iMoonLab/HGNN

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
    "auc", "auprc", "accuracy", "precision", "recall", "f1",
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


# Torch device from "auto", "cpu" or "cuda" ("auto" = cuda when available).
def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


# Metrics of the dropout class (label 1) at threshold 0.5: auc, auprc, accuracy, precision,
# recall, f1.
def metrics(labels, probabilities):
    predicted = probabilities >= config.THRESHOLD                       # ŷ = 1[p ≥ 0.5]
    # PRECISION = TP/(TP + FP)
    # RRCALL = TP/(TP + FN)
    # F1 = 2·P·R/(P + R)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=[1], zero_division=0)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),             # AUC = P(p_dropout > p_non-dropout)
        "auprc": float(average_precision_score(labels, probabilities)), # AUPRC = Σ_n (R_n − R_{n−1})·P_n
        "accuracy": float(np.mean(predicted == labels)),                # Acc = (TP + TN)/n
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
    }


# Dropout probabilities
def predict(model, x, graph, index):
    was_training = model.training
    model.eval()                                                         # no dropout
    with torch.no_grad():                                                # no gradients while scoring
        logits = model(x, graph)                                         # every node of H
        probabilities = torch.sigmoid(logits[index])                     # p = σ(logit) of the chosen nodes
    model.train(was_training)
    return probabilities.cpu().numpy()


# Append one row to outputs/results.csv (header written when the file is new).
def append_result(row):
    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    is_new = not config.RESULTS_CSV.exists()
    with open(config.RESULTS_CSV, "a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS)
        if is_new:
            writer.writeheader()
        # Floats are written with 6 decimals, everything else as it is.
        text_row = {}
        for name in row:
            value = row[name]
            if isinstance(value, float):
                text_row[name] = f"{value:.6f}"
            else:
                text_row[name] = value
        writer.writerow(text_row)
    log("result", f"appended to {config.RESULTS_CSV}")


# Train
def train_one_seed(code, seed, settings, data, device):
    started_at = time.perf_counter()
    scenario = config.SCENARIOS[code]
    set_seed(seed)
    x = torch.as_tensor(data["features"], device=device)                     # X of every node of H
    graph = prepare_graph(data["graph"], device)
    index = {}
    split_labels = {}
    for split in ("train", "validation", "test"):
        index[split] = torch.as_tensor(data[f"{split}_index"], device=device)
        split_labels[split] = data["labels"][data[f"{split}_index"]]       # NumPy, read by metrics
    # Only the train labels go to the device: the loss sees no validation or test label.
    train_labels = torch.as_tensor(split_labels["train"], device=device)

    model = DropoutModel(x.shape[1], settings["hidden_dim"], settings["dropout"],
                         hgnn_layers=scenario["hgnn_layers"], use_mlp=scenario["use_mlp"],
                         learn_w=scenario["learn_w"])
    model = model.to(device)
    family_logits = model.hgnn.family_logits

    # Every weight except the family weights W: Θ, A, u.
    other_parameters = []
    for parameter in model.parameters():
        if parameter is not family_logits:
            other_parameters.append(parameter)

    # Adam with L2 (weight_decay λ): g = ∇L + λ·θ, then θ ← θ − lr·m̂/(√v̂ + ε)
    groups = [{"params": other_parameters}]                                          # Θ, A, u: lr, L2
    if scenario["learn_w"]:                                                          # W: own lr, no L2
        groups.append({"params": [family_logits], "lr": settings["family_weight_lr"], "weight_decay": 0.0})
    optimizer = torch.optim.Adam(groups, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])

    best = {"auc": -1.0, "epoch": 0, "state": None}
    stale = 0          # validations in a row without a better AUC
    epoch = 0
    for epoch in range(1, settings["epochs"] + 1):
        # One full-batch training step: forward over the whole H, loss on the train nodes only.
        model.train()
        optimizer.zero_grad()
        logits = model(x, graph)                                         # every node of H
        train_logits = logits[index["train"]]
        # L = −(1/|V_train|)·Σ_{v∈V_train} [y·log p + (1−y)·log(1−p)]
        loss = F.binary_cross_entropy_with_logits(train_logits, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)          # ‖g‖ ≤ 5
        optimizer.step()

        if epoch % 50 == 0:
            train_auc = roc_auc_score(split_labels["train"], train_logits.detach().cpu().numpy())
            log("train", f"{code} seed {seed} epoch {epoch}: loss={loss.item():.4f}, train_auc={train_auc:.4f}")

        # Early stopping: validation AUC every eval_every epochs, keep the best weights in memory.
        if epoch % settings["eval_every"] == 0:
            validation_probabilities = predict(model, x, graph, index["validation"])
            validation_auc = roc_auc_score(split_labels["validation"], validation_probabilities)
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
        probabilities = predict(model, x, graph, index[split])
        scores[split] = metrics(split_labels[split], probabilities)
        parts = []
        for name in scores[split]:
            parts.append(f"{name}={scores[split][name]:.4f}")
        log(split, f"{code} seed {seed}: " + ", ".join(parts))

    # One row of results.csv.
    row = {}
    row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    row["scenario"] = code
    row["seed"] = seed
    row["families"] = "+".join(scenario["families"])
    row["features"] = scenario["features"]
    row["hgnn_layers"] = scenario["hgnn_layers"]
    row["use_mlp"] = scenario["use_mlp"]
    row["learn_w"] = scenario["learn_w"]
    for name in ("epochs", "eval_every", "patience", "hidden_dim", "dropout", "learning_rate", "weight_decay"):
        row[name] = settings[name]
    row["best_epoch"] = best["epoch"]
    row["epochs_run"] = epoch
    for name in ("auc", "auprc", "f1"):
        row[f"val_{name}"] = scores["validation"][name]
    for name in scores["test"]:
        row[f"test_{name}"] = scores["test"][name]
    weights = model.weight_summary(scenario["families"])
    for name in weights:
        row[name] = weights[name]
    row["minutes"] = (time.perf_counter() - started_at) / 60
    append_result(row)


# Scenario codes to run: --scenario all -> config.SCENARIOS_ALL; otherwise the given codes, checked.
def scenario_codes(values):
    if values == ["all"]:
        codes = list(config.SCENARIOS_ALL)
    else:
        codes = values
    unknown = []
    for code in codes:
        if code not in config.SCENARIOS:
            unknown.append(code)
    if unknown:
        raise SystemExit(f"unknown scenarios {unknown}; choose from {list(config.SCENARIOS)} or 'all'")
    return codes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment scenarios: train with early stopping "
                                                 "on validation AUC, then score validation and test at 0.5.")
    all_codes = " ".join(config.SCENARIOS_ALL)
    parser.add_argument("--scenario", nargs="+", default=["M"],
                        help=f"'all' ({all_codes}) or codes from {list(config.SCENARIOS)}")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    parser.add_argument("--epochs", type=int, default=config.TRAIN["epochs"])
    parser.add_argument("--eval-every", type=int, default=config.TRAIN["eval_every"])
    parser.add_argument("--patience", type=int, default=config.TRAIN["patience"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default=config.PROCESSED)
    arguments = parser.parse_args()

    codes = scenario_codes(arguments.scenario)
    settings = dict(config.TRAIN)
    settings["epochs"] = arguments.epochs
    settings["eval_every"] = arguments.eval_every
    settings["patience"] = arguments.patience
    device = resolve_device(arguments.device)
    log("setup", f"device={device}, scenarios={codes}, seeds={arguments.seeds}, settings={settings}")

    # Read the data once (all families, all columns); every scenario takes its part of it.
    loaded = graph_data.load_graph(arguments.output_dir)
    split_counts = []
    for split_id in range(len(config.SPLITS)):
        count = int(np.sum(loaded["split"] == split_id))
        split_counts.append(f"{config.SPLITS[split_id]}={count:,}")
    log("setup", f"H over {loaded['graph']['num_nodes']:,} nodes: " + ", ".join(split_counts))

    # Scenario by scenario; every seed of a scenario before the next one.
    for code in codes:
        scenario = config.SCENARIOS[code]   # the Vietnamese description is not logged (Windows console)
        families_text = "+".join(scenario["families"])
        log("scenario", f"{code}: families={families_text}, features={scenario['features']}, "
                        f"hgnn_layers={scenario['hgnn_layers']}, use_mlp={scenario['use_mlp']}, "
                        f"learn_w={scenario['learn_w']}")
        data = graph_data.apply_scenario(loaded, config.SCENARIOS[code])
        for seed in arguments.seeds:
            train_one_seed(code, seed, settings, data, device)
