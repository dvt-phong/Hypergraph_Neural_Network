# 10. Summarize outputs/oulad/summary.csv, outputs/oulad/ket_qua.xlsx

import sys
from importlib import import_module

import pandas as pd

config = import_module("0_config")

METRICS = ("val_auc", "val_auprc", "val_f1",
           "test_auc", "test_auprc", "test_accuracy", "test_precision", "test_recall", "test_f1")
WEIGHTS = ("w_course", "w_object", "w_user", "w_self_loop")
COMPONENTS = ("course", "object", "user", "self_loop")


# Read results.csv and write summary.csv and ket_qua.xlsx (one row per scenario, mean ± std).
def summarize():
    runs = pd.read_csv(config.RESULTS_CSV)
    runs = runs.drop_duplicates(["scenario", "seed"], keep="last")

    order = {}
    index = 0
    for code in config.SCENARIOS:
        order[code] = index
        index += 1
    runs["order"] = runs["scenario"].map(order)
    runs = runs.sort_values(["order", "seed"])
    runs = runs.drop(columns="order")
    
    weights = pd.DataFrame(index=runs.index)
    for name in WEIGHTS:
        weights[name] = pd.to_numeric(runs[name], errors="coerce")                # "" -> NaN
    for name in WEIGHTS:
        runs[f"{name}_share"] = weights[name] / weights.sum(axis=1)                # w_f / Σ_f w_f

    rows = []
    for code, group in runs.groupby("scenario", sort=False):
        if code in config.SCENARIOS:
            description = config.SCENARIOS[code]["description"]
        else:
            description = ""                                                       # scenario no longer in config
        families = str(group["families"].iloc[-1]).split("+")
        row = {}
        row["scenario"] = code
        row["description"] = description
        for name in COMPONENTS:                                                    # ✓ / ✗ columns
            if name in families:
                row[name] = "✓"
            else:
                row[name] = "✗"
        row["features"] = group["features"].iloc[-1]
        row["hgnn_layers"] = group["hgnn_layers"].iloc[-1]
        if bool(group["use_mlp"].iloc[-1]):
            row["mlp"] = "✓"
        else:
            row["mlp"] = "✗"
        if bool(group["learn_w"].iloc[-1]):
            row["learn_w"] = "✓"
        else:
            row["learn_w"] = "✗"
        seed_texts = []
        for seed in group["seed"]:
            seed_texts.append(str(seed))
        row["seeds"] = " ".join(seed_texts)
        row["n"] = len(group)
        for name in METRICS:
            # mean = (1/n)·Σ x,  std = √(Σ(x − mean)²/(n − 1))  over the n seeds
            mean = group[name].mean()
            std = group[name].std(ddof=1)
            if pd.isna(std):
                std = 0.0                                                          # one seed: no std
            row[name] = f"{mean:.4f} ± {std:.4f}"
        row["test_auc_mean"] = group["test_auc"].mean()
        row["best_epoch"] = round(group["best_epoch"].mean())
        for name in WEIGHTS:
            share = group[f"{name}_share"].mean()
            if pd.isna(share):
                row[f"{name}_share"] = ""
            else:
                row[f"{name}_share"] = round(share, 4)
        row["minutes"] = round(group["minutes"].mean(), 1)
        rows.append(row)
    summary = pd.DataFrame(rows)

    # Δ test AUC = mean test AUC of the scenario − mean test AUC of M.
    if "M" in summary["scenario"].tolist():
        reference = summary.loc[summary["scenario"] == "M", "test_auc_mean"].iloc[0]
        summary.insert(summary.columns.get_loc("test_auc") + 1, "delta_test_auc_vs_M",
                       (summary["test_auc_mean"] - reference).round(4))
    summary = summary.drop(columns="test_auc_mean")

    summary.to_csv(config.OUTPUTS / "summary.csv", index=False, encoding="utf-8-sig")   # BOM: Excel shows "±", "✓"
    with pd.ExcelWriter(config.OUTPUTS / "ket_qua.xlsx") as writer:
        summary.to_excel(writer, sheet_name="Tong_hop", index=False)
        runs.to_excel(writer, sheet_name="Tung_seed", index=False)
    print(summary.drop(columns="description").to_string(index=False))
    print(f"Saved {config.OUTPUTS / 'summary.csv'} and {config.OUTPUTS / 'ket_qua.xlsx'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # print "±" and "✓" also on a Windows console
    summarize()
