# Plots for one scripts/run_all.sh run folder (after collect_results.py):
#
#   python scripts/plot_results.py result/<dd-mm-yyyy_HH-MM>
#
# Writes into the run folder:
#   history.png        by epoch, one line per seed: train BCE, train and validation
#                      AUC, the share of H0 memberships HSL keeps, and the best-F1
#                      threshold of the validation subset (a dot marks the best epoch)
#   probabilities.png  one panel per seed: validation probabilities of dropouts and
#                      non-dropouts, with the chosen threshold t* and 0.5
# Needs matplotlib (requirements.txt).

import json
import sys
from importlib import import_module
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
SEEDS = import_module("0_config").SEEDS

# Categorical slots 1-5 of the validated default palette, one per seed of
# config.SEEDS, so a seed has the same color in every figure and run.
SEED_COLORS = dict(zip(SEEDS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")))
CLASS_COLORS = {1: "#eb6834", 0: "#2a78d6"}  # dropout, non-dropout
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
KEPT_STYLES = {"kept_course": "-", "kept_object": "--", "kept_behavioral": ":"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def load_reports(reports_dir):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in reports_dir.glob("*_train.json")]
    return sorted(reports, key=lambda report: report["seed"])


def plot_history(reports, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    loss_axis, auc_axis, kept_axis, threshold_axis = axes.flat
    for report in reports:
        history = report["history"]
        if not history:
            continue
        color = SEED_COLORS.get(report["seed"], MUTED)
        label = f"seed {report['seed']}"
        epochs = [record["epoch"] for record in history]
        validated = [record for record in history if "validation" in record]
        validated_epochs = [record["epoch"] for record in validated]

        loss_axis.plot(epochs, [record["bce"] for record in history], color=color, label=label)
        auc_axis.plot(epochs, [record["train_auc"] for record in history], color=color, linestyle="--",
                      linewidth=1.2)
        auc_axis.plot(validated_epochs, [record["validation"]["auc"] for record in validated],
                      color=color, label=label)
        best = next((record for record in validated if record["epoch"] == report["best_epoch"]), None)
        if best is not None:
            auc_axis.plot(best["epoch"], best["validation"]["auc"], "o", color=color, markersize=8,
                          markeredgecolor=SURFACE, markeredgewidth=2)
        for key, style in KEPT_STYLES.items():
            if key in history[0]:
                kept_axis.plot(epochs, [record[key] for record in history], color=color, linestyle=style,
                               label=f"{label}, {key.removeprefix('kept_')}")
        thresholds = [record["validation"].get("threshold") for record in validated]
        if any(value is not None for value in thresholds):
            threshold_axis.plot(validated_epochs, thresholds, color=color, label=label)

    loss_axis.set(title="BCE trên train", xlabel="epoch")
    auc_axis.set(title="AUC: validation (liền), train (đứt); chấm = best epoch", xlabel="epoch")
    kept_axis.set(title="Membership H0 được giữ\n(course liền, object đứt, behavioral chấm)",
                  xlabel="epoch", ylim=(0, 1.02))
    threshold_axis.set(title="Ngưỡng tối ưu F1 trên tập con validation", xlabel="epoch")
    threshold_axis.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    for axis in (loss_axis, auc_axis, threshold_axis):
        if axis.get_lines():
            axis.legend(frameon=False, fontsize=9)
    if not kept_axis.get_lines():
        kept_axis.text(0.5, 0.5, "không dùng HSL", ha="center", va="center", color=MUTED,
                       transform=kept_axis.transAxes)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved {output_path}")


def plot_probabilities(reports_dir, reports, output_path):
    panels = []
    for report in reports:
        run_name = Path(report["checkpoint"]).stem
        path = reports_dir / f"{run_name}_validation_probs.npz"
        if path.exists():
            with np.load(path) as saved:
                panels.append((report["seed"], saved["labels"], saved["probabilities"], float(saved["threshold"])))
    if not panels:
        print("No *_validation_probs.npz found; skipped probabilities.png")
        return

    figure, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.4), sharey=True,
                                constrained_layout=True, squeeze=False)
    bins = np.linspace(0, 1, 51)
    for axis, (seed, labels, probabilities, threshold) in zip(axes[0], panels):
        for label, name in ((1, "bỏ học"), (0, "không bỏ")):
            axis.hist(probabilities[labels == label], bins=bins, density=True, histtype="step",
                      linewidth=2, color=CLASS_COLORS[label], label=name)
        axis.axvline(threshold, color=INK, linewidth=1.5)
        axis.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
        # Label on the side away from the 0.5 line so the two never overlap.
        left = threshold < 0.5
        axis.text(threshold, 0.98, f"t*={threshold:.2f} " if left else f" t*={threshold:.2f}",
                  ha="right" if left else "left", va="top", fontsize=9,
                  transform=axis.get_xaxis_transform())
        axis.set(title=f"seed {seed}", xlabel="xác suất bỏ học (validation)", xlim=(0, 1))
    axes[0][0].set_ylabel("mật độ")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved {output_path}")


def main(run_dir):
    run_dir = Path(run_dir)
    reports_dir = run_dir / "reports" if (run_dir / "reports").is_dir() else run_dir
    reports = load_reports(reports_dir)
    if not reports:
        sys.exit(f"No *_train.json in {reports_dir}")
    plot_history(reports, run_dir / "history.png")
    plot_probabilities(reports_dir, reports, run_dir / "probabilities.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/plot_results.py result/<run folder>")
    main(sys.argv[1])
