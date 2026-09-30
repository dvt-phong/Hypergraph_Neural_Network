# One detailed results table over every run folder in result/.
#
#   python scripts/summarize_results.py              reads result/*, writes into result/
#   python scripts/summarize_results.py --result-dir path/to/result
#
# Every metric is recomputed from the saved probabilities (*_validation_probs.npz,
# *_test_probs.npz) at the threshold t* stored with them, so old runs get the
# metrics added later (accuracy, specificity) and all rows are computed the same
# way. A run without probability files falls back to the numbers in its reports.
#
# Output:
#   tong_hop_ket_qua.xlsx   sheets Lan_chay (one row per run folder, mean ± std over
#                           its seeds), Kich_ban (one row per configuration, the
#                           latest run of each seed, so seeds run in separate
#                           folders are merged), Theo_seed (one row per seed), Giai_thich
#   tong_hop_lan_chay.csv, tong_hop_kich_ban.csv, tong_hop_theo_seed.csv
#   tong_hop_ket_qua.md     short Markdown table of the test metrics per configuration

import argparse
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

# (key, header): metrics computed for validation and test.
METRICS = (
    ("auroc", "AUROC (AUC)"),
    ("auprc", "AUPRC"),
    ("accuracy", "ACC"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("specificity", "Specificity"),
    ("macro_f1", "Macro-F1"),
    ("f1_negative", "F1 lớp không bỏ"),
    ("auprc_negative", "AUPRC lớp không bỏ"),
)
# (key in the train settings, header)
HYPERPARAMETERS = (
    ("families", "Hyperedge"),
    ("hypergraph", "File graph"),
    ("skip_connection", "Skip"),
    ("family_weights", "W family"),
    ("edge_weights", "W từng hyperedge"),
    ("edge_sampling", "Me"),
    ("node_sampling", "Mv"),
    ("add_per_edge", "ΔH/cạnh"),
    ("lambda_cl", "λ contrastive"),
    ("temperature", "τ contrastive"),
    ("hidden_dim", "Chiều ẩn"),
    ("dropout", "Dropout"),
    ("learning_rate", "LR"),
    ("weight_decay", "Weight decay"),
    ("family_weight_lr", "LR W family"),
    ("edge_weight_lr", "LR W hyperedge"),
    ("epochs", "Epoch tối đa"),
    ("patience", "Patience"),
    ("eval_every", "Validate mỗi"),
    ("validation_limit", "Tập con val"),
    ("select_metric", "Chọn checkpoint"),
    ("pos_weight", "pos_weight"),
    ("lr_schedule", "LR schedule"),
    ("feature_set", "Feature"),
)
# Learned structure at the best epoch (HSL keep rates, W).
STRUCTURE = ("kept_course", "kept_object", "kept_behavioral", "kept_user",
             "w_course", "w_object", "w_behavioral", "w_user", "w_self_loop",
             "alpha_course", "alpha_object", "alpha_behavioral", "alpha_user")


def metrics_at(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1], zero_division=0
    )
    return {
        "auroc": roc_auc_score(labels, probabilities),
        "auprc": average_precision_score(labels, probabilities),
        "accuracy": float(np.mean(predicted == labels)),
        "precision": precision[1],
        "recall": recall[1],
        "f1": f1[1],
        "specificity": recall[0],
        "macro_f1": f1.mean(),
        "f1_negative": f1[0],
        "auprc_negative": average_precision_score(1 - labels, 1 - probabilities),
    }


# Metrics of one split from the probability file, else from the report.
def split_metrics(reports, run_name, split, fallback):
    path = reports / f"{run_name}_{split}_probs.npz"
    if path.exists():
        with np.load(path) as saved:
            labels = saved["labels"].astype(int)
            return metrics_at(labels, saved["probabilities"], float(saved["threshold"]))
    fallback = fallback or {}
    return {key: fallback.get("auc" if key == "auroc" else key) for key, _ in METRICS}


def read_run_info(run_dir):
    info = {}
    path = run_dir / "run_info.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
    return info


def read_durations(run_dir):
    path = run_dir / "manifest.tsv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as manifest:
        return {int(row["seed"]): int(row["duration_sec"]) for row in csv.DictReader(manifest, delimiter="\t")}


def folder_time(run_dir):
    try:
        return datetime.strptime(run_dir.name[:16], "%d-%m-%Y_%H-%M")
    except ValueError:
        return datetime.fromtimestamp(run_dir.stat().st_mtime)


def model_name(settings):
    if settings.get("model") in ("gbdt", "logreg"):
        return {"gbdt": "GBDT", "logreg": "LR"}[settings["model"]]
    families = [f for f in settings.get("families", ["course", "object", "behavioral"]) if f != "self_loop"]
    if not families and not settings.get("hsl", True):
        return "MLP"
    return "HSL" if settings.get("hsl", True) else "HGNN"


def seed_rows(run_dir):
    reports = run_dir / "reports"
    info = read_run_info(run_dir)
    durations = read_durations(run_dir)
    rows = []
    for train_path in sorted(reports.glob("*_train.json")):
        run_name = train_path.name.removesuffix("_train.json")
        train = json.loads(train_path.read_text(encoding="utf-8"))
        test_path = reports / f"{run_name}_test.json"
        test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {}
        settings = train.get("settings", {})
        best = next((r for r in train.get("history", []) if r.get("epoch") == train.get("best_epoch")), {})
        row = {
            "folder": run_dir.name,
            "started": folder_time(run_dir),
            "note": info.get("note", ""),
            "git": info.get("git", ""),
            "scenario": re.sub(r"_seed_\d+$", "", run_name),
            "tag": settings.get("tag", ""),
            "model": model_name(settings),
            "seed": train.get("seed"),
            "best_epoch": train.get("best_epoch"),
            "epochs_run": len(train.get("history", [])) or None,
            "threshold": train.get("threshold"),
            "duration_min": round(durations[train["seed"]] / 60, 1) if train.get("seed") in durations else None,
            "settings": settings,
            "structure": {key: best.get(key) for key in STRUCTURE},
        }
        validation = split_metrics(reports, run_name, "validation", train.get("best_validation"))
        tested = split_metrics(reports, run_name, "test", test.get("test"))
        row.update({f"val_{key}": value for key, value in validation.items()})
        row.update({f"test_{key}": value for key, value in tested.items()})
        rows.append(row)
    return rows


def mean_std(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def text(value):
    if isinstance(value, bool):
        return "Có" if value else "Không"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value if item != "self_loop")
    return "" if value is None else value


# One summary row over a group of seed rows (one folder, or one configuration).
def summarize(group, first_columns):
    last = group[-1]
    row = dict(first_columns)
    row["Mô hình"] = last["model"]
    row["Kịch bản (tag)"] = last["tag"]
    row["Seeds"] = " ".join(str(r["seed"]) for r in sorted(group, key=lambda r: r["seed"]))
    row["Số seed"] = len(group)
    for key, header in HYPERPARAMETERS:
        row[header] = text(last["settings"].get(key))
    for split, split_name in (("test", "test"), ("val", "val")):
        for key, header in METRICS:
            mean, std = mean_std(r[f"{split}_{key}"] for r in group)
            row[f"{header} {split_name}"] = "" if mean is None else f"{mean:.4f} ± {std:.4f}"
    for column, key in (("Ngưỡng t*", "threshold"), ("Best epoch", "best_epoch"),
                        ("Epoch đã chạy", "epochs_run"), ("Phút/seed", "duration_min")):
        mean, std = mean_std(r[key] for r in group)
        row[column] = "" if mean is None else (f"{mean:.4f}" if key == "threshold" else f"{mean:.0f}")
    for key in STRUCTURE:
        mean, _ = mean_std(r["structure"][key] for r in group)
        row[key] = "" if mean is None else round(mean, 4)
    row["_sort"] = mean_std(r["test_auroc"] for r in group)[0] or 0.0
    return row


def seed_table(rows):
    table = []
    for r in rows:
        row = {"Lần chạy": r["folder"], "Kịch bản (tag)": r["tag"], "Cấu hình": r["scenario"],
               "Mô hình": r["model"], "Seed": r["seed"], "Best epoch": r["best_epoch"],
               "Epoch đã chạy": r["epochs_run"], "Ngưỡng t*": r["threshold"], "Phút": r["duration_min"]}
        for split in ("test", "val"):
            for key, header in METRICS:
                value = r[f"{split}_{key}"]
                row[f"{header} {split}"] = None if value is None else round(float(value), 4)
        row.update({key: r["structure"][key] for key in STRUCTURE})
        table.append(row)
    return table


def write_csv(path, table):
    columns = [c for c in table[0] if not c.startswith("_")] if table else []
    with open(path, "w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)


GUIDE = (
    ("Lan_chay", "Mỗi dòng là một thư mục result/<ngày_giờ> (một lần gọi run_all.sh), mean ± std trên các seed của nó."),
    ("Kich_ban", "Mỗi dòng là một cấu hình (tên run bỏ phần seed). Với mỗi seed lấy lần chạy mới nhất, nên seed 1 chạy "
                 "lúc sàng lọc và seed 11–11111 chạy sau được gộp thành 5 seed."),
    ("Theo_seed", "Từng seed, số chưa làm tròn thành mean ± std."),
    ("AUROC (AUC)", "Diện tích dưới đường ROC. AUC và AUROC là cùng một chỉ số. Không phụ thuộc ngưỡng."),
    ("AUPRC", "Diện tích dưới đường Precision–Recall của lớp bỏ học (average precision)."),
    ("ACC", "Tỉ lệ dự đoán đúng tại ngưỡng t*. Lớp bỏ học chiếm 75,8%, nên đoán 'ai cũng bỏ' đã đạt ACC 0,758."),
    ("Precision, Recall, F1", "Của lớp bỏ học (nhãn 1), tại ngưỡng t*."),
    ("Specificity", "Recall của lớp không bỏ học: tỉ lệ học viên không bỏ được nhận ra đúng."),
    ("Macro-F1", "Trung bình F1 của hai lớp."),
    ("Ngưỡng t*", "Chọn trên toàn bộ validation để F1 lớp bỏ học cao nhất, rồi dùng nguyên cho test."),
    ("kept_*", "HSL: tỉ lệ membership được giữ lại theo loại hyperedge, ở best epoch."),
    ("w_*", "Trọng số family học được (--family-weights), ở best epoch. Chỉ tỉ lệ giữa các w có nghĩa."),
    ("alpha_*", "Trung bình trọng số từng hyperedge α_e theo loại (--edge-weights). 1 = chưa khác mặc định."),
    ("Chọn mô hình", "So sánh và chọn bằng cột val; cột test chỉ để báo cáo."),
)


def write_xlsx(path, sheets):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, table in sheets:
        sheet = workbook.create_sheet(name)
        columns = [c for c in table[0] if not c.startswith("_")] if table else []
        sheet.append(columns)
        for row in table:
            sheet.append([row.get(column) for column in columns])
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        sheet.freeze_panes = "B2" if name != "Giai_thich" else None
        for index, column in enumerate(columns, 1):
            width = max([len(str(column))] + [len(str(row.get(column) or "")) for row in table[:200]])
            sheet.column_dimensions[sheet.cell(row=1, column=index).column_letter].width = min(max(10, width + 2), 60)
    workbook.save(path)


def write_markdown(path, table):
    columns = ["Cấu hình", "Mô hình", "Số seed", "AUROC (AUC) test", "AUPRC test", "ACC test",
               "Precision test", "Recall test", "F1 test", "Macro-F1 test", "AUROC (AUC) val"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in table:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(result_dir):
    result_dir = Path(result_dir)
    folders = sorted((d for d in result_dir.iterdir() if (d / "reports").is_dir()), key=folder_time)
    rows = []
    for folder in folders:
        rows.extend(seed_rows(folder))
    if not rows:
        raise SystemExit(f"No run with reports under {result_dir}")

    by_folder = {}
    for r in rows:
        by_folder.setdefault(r["folder"], []).append(r)
    runs = []
    for number, (folder, group) in enumerate(by_folder.items(), 1):
        runs.append(summarize(group, {"STT": number, "Lần chạy": folder,
                                      "Ngày": group[0]["started"].strftime("%d/%m/%Y %H:%M"),
                                      "Cấu hình": group[-1]["scenario"], "Ghi chú": group[-1]["note"],
                                      "Git": group[-1]["git"]}))

    latest = {}
    for r in rows:  # folders are in time order, so later runs of a seed replace earlier ones
        latest[(r["scenario"], r["seed"])] = r
    by_scenario = {}
    for (scenario, _), r in latest.items():
        by_scenario.setdefault(scenario, []).append(r)
    scenarios = [summarize(group, {"Cấu hình": scenario,
                                   "Lần chạy": ", ".join(sorted({r["folder"] for r in group}))})
                 for scenario, group in by_scenario.items()]
    scenarios.sort(key=lambda row: -row["_sort"])

    seeds = seed_table(rows)
    write_csv(result_dir / "tong_hop_lan_chay.csv", runs)
    write_csv(result_dir / "tong_hop_kich_ban.csv", scenarios)
    write_csv(result_dir / "tong_hop_theo_seed.csv", seeds)
    write_markdown(result_dir / "tong_hop_ket_qua.md", scenarios)
    try:
        guide = [{"Mục": name, "Giải thích": meaning} for name, meaning in GUIDE]
        write_xlsx(result_dir / "tong_hop_ket_qua.xlsx",
                   [("Kich_ban", scenarios), ("Lan_chay", runs), ("Theo_seed", seeds), ("Giai_thich", guide)])
    except ImportError:
        print("openpyxl not installed; wrote CSV and Markdown only")
    print(f"{len(runs)} run folders, {len(scenarios)} configurations, {len(seeds)} seeds -> {result_dir}/tong_hop_*")
    for row in scenarios[:15]:
        print(f"  {row['Cấu hình']:<55} n={row['Số seed']}  test AUROC {row['AUROC (AUC) test']}  "
              f"val AUROC {row['AUROC (AUC) val']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize every run folder in result/.")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    main(parser.parse_args().result_dir)
