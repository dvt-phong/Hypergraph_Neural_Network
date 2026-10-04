# 10. Summarize outputs/results.csv: one row per scenario with mean ± std over its seeds,
#     in the order of config.SCENARIOS, plus Δ test AUC against the main model M.
#     When a (scenario, seed) pair was run more than once, only its latest row counts.
#
#   python src/10_summary.py   -> outputs/summary.csv, outputs/ket_qua.xlsx

import sys
from importlib import import_module

import pandas as pd

config = import_module("0_config")

METRICS = ("val_auc", "val_auprc", "val_f1",
           "test_auc", "test_auprc", "test_accuracy", "test_precision", "test_recall", "test_f1")
WEIGHTS = ("w_course", "w_object", "w_user", "w_self_loop")
COMPONENTS = ("course", "object", "user", "self_loop")


def summarize():
    runs = pd.read_csv(config.RESULTS_CSV)
    runs = runs.drop_duplicates(["scenario", "seed"], keep="last")
    order = {code: index for index, code in enumerate(config.SCENARIOS)}          # table order
    runs = runs.sort_values(["scenario", "seed"], key=lambda column: column.map(order)
                            if column.name == "scenario" else column)
    # Only the ratios of the family weights matter: G is unchanged when every w is scaled.
    # Families a scenario does not use are empty and left out of the sum.
    weights = runs[list(WEIGHTS)].apply(pd.to_numeric, errors="coerce")
    for name in WEIGHTS:
        runs[f"{name}_share"] = weights[name] / weights.sum(axis=1)                # w_f / Σ_f w_f

    rows = []
    for code, group in runs.groupby("scenario", sort=False):
        scenario = config.SCENARIOS.get(code, {})
        families = str(group["families"].iloc[-1]).split("+")
        row = {"scenario": code, "description": scenario.get("description", "")}
        for name in COMPONENTS:                                                    # ✓ / ✗ columns
            row[name] = "✓" if name in families else "✗"
        row.update({"features": group["features"].iloc[-1],
                    "hgnn_layers": group["hgnn_layers"].iloc[-1],
                    "mlp": "✓" if bool(group["use_mlp"].iloc[-1]) else "✗",
                    "seeds": " ".join(str(seed) for seed in group["seed"]), "n": len(group)})
        for name in METRICS:
            mean, std = group[name].mean(), group[name].std(ddof=1)               # std = √(Σ(x − mean)²/(n − 1))
            row[name] = f"{mean:.4f} ± {0.0 if pd.isna(std) else std:.4f}"
        row["test_auc_mean"] = group["test_auc"].mean()
        row["best_epoch"] = round(group["best_epoch"].mean())
        for name in WEIGHTS:
            share = group[f"{name}_share"].mean()
            row[f"{name}_share"] = "" if pd.isna(share) else round(share, 4)
        row["minutes"] = round(group["minutes"].mean(), 1)
        rows.append(row)
    summary = pd.DataFrame(rows)

    # Δ test AUC = mean test AUC of the scenario − mean test AUC of M.
    if "M" in set(summary["scenario"]):
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
