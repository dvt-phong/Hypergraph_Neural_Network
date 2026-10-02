# The experiment scenarios of the paper, in one place.
#
#   python scripts/scenarios.py list            code, group, description, question
#   python scripts/scenarios.py codes           every code, in table order
#   python scripts/scenarios.py args M0         training script, then its arguments, one per line
#   python scripts/scenarios.py note M0         one-line note for the workbook
#   python scripts/scenarios.py check M0 [...]  the hypergraph bundle exists and holds the
#                                               User rule the scenario needs; the
#                                               baseline's code and environment exist
#   python scripts/scenarios.py python CODE     virtual environment of the scenario
#                                               (.venvs/<name>; empty = the main one)
#   python scripts/scenarios.py bundle CODE     hypergraph bundle the scenario reads
#   python scripts/scenarios.py cache CODE [...] exit 0 if one of them needs the event
#                                               cache of src/10_baseline_data.py
#
# scripts/run_scenarios.sh runs them; scripts/summarize_results.py lists them in
# result/so_thi_nghiem.xlsx (sheets Kich_ban and Bang_chinh). The scenario code
# is also the run tag (--tag), so every report says which scenario it belongs to.
#
# Hyperparameters not set here are the defaults of DEFAULT_SETTINGS in
# src/8_train.py: hidden 128, dropout 0.5, lr 1e-3, weight decay 5e-4, family
# weight lr 0.05 without weight decay, validation every 5 epochs, checkpoint on
# validation AUPRC, pos_weight 1, t* on the full validation split. The
# workbook records the full settings of every run.
#
# User hyperedges come from two bundles in data/processed/simple:
#   hypergraph.npz           rule "any": all enrollments of the learner, also in
#                            courses that start later (uses future information,
#                            like the baselines as published)
#   hypergraph_temporal.npz  rule "temporal": only courses that started no later;
#                            with --causal nothing reaches a node from after its
#                            35-day window (scripts/check_leakage.py)

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "simple"
TRAIN = "src/8_train.py"
BASELINES = "src/9_baselines.py"
# Published baselines (GĐ2), same split and protocol: src/10_baseline_data.py.
# The ones that need other libraries run in their own environment,
# made by scripts/setup_baselines.sh.
HYPERGCN = "src/11_hypergcn.py"
SIGNET = "src/12_signet.py"
MSTGCN = "src/13_mstgcn.py"
CATFHN = "src/14_catfhn.py"
CFIN = "src/15_cfin.py"
CACHE = PROCESSED / "baselines" / "vocab.npz"

LONG = ["--epochs", "1000", "--patience", "60"]
MAIN = ["--no-hsl", "--skip-connection", "--family-weights"]  # HGNN + skip MLP(X) + W per family
ANY = ["--hypergraph", "hypergraph.npz"]
TEMPORAL = ["--hypergraph", "hypergraph_temporal.npz", "--causal"]


def scenario(group, description, question, args, *, script=TRAIN, user_rule=None,
             bundle=None, venv=None, repo=None, cache=False):
    return {"group": group, "description": description, "question": question,
            "script": script, "args": args, "user_rule": user_rule,
            "bundle": bundle, "venv": venv, "repo": repo, "cache": cache}


# One published baseline under both User rules: <code> without future
# information (compare with M0), <code>-any as published (compare with M-any).
def published(code, name, question, script, *, venv=None, repo=None, graph=False):
    entries = {}
    for rule, suffix, note in (("temporal", "", "không tương lai, so với M0"),
                               ("any", "-any", "có khoá học tương lai như bản công bố, so với M-any")):
        entries[code + suffix] = scenario(
            "Baseline", f"{name}, User {rule} ({note})", question, ["--user-rule", rule],
            script=script, user_rule=rule if graph else None,
            bundle=("hypergraph_temporal.npz" if rule == "temporal" else "hypergraph.npz") if graph else None,
            venv=venv, repo=repo, cache=not graph)
    return entries


SCENARIOS = {
    "M-any": scenario(
        "Chính", "HGNN + skip MLP + W; Course, Object, User (any), self-loop",
        "Cùng điều kiện với baseline (User có cả khoá học tương lai); tái hiện 0,8749 (u1-hgnn)",
        MAIN + ["--families", "course,object,user"] + ANY + LONG, user_rule="any"),
    "M0": scenario(
        "Chính", "Như M-any, User temporal + causal",
        "Kết quả chính không rò rỉ thông tin tương lai",
        MAIN + ["--families", "course,object,user"] + TEMPORAL + LONG, user_rule="temporal"),
    "A1": scenario(
        "Ablation", "M0 bỏ User (Course, Object)", "User đóng góp bao nhiêu khi không rò rỉ",
        MAIN + ["--families", "course,object"] + LONG),
    "A2": scenario(
        "Ablation", "M0 bỏ Object (Course, User temporal)", "Vai trò của Object",
        MAIN + ["--families", "course,user"] + TEMPORAL + LONG, user_rule="temporal"),
    "A3": scenario(
        "Ablation", "M0 bỏ Course (Object, User temporal)", "Vai trò của Course",
        MAIN + ["--families", "object,user"] + TEMPORAL + LONG, user_rule="temporal"),
    "A4": scenario(
        "Ablation", "M0 bỏ W (mọi loại hyperedge cùng trọng số)", "W theo loại hyperedge có cần không",
        ["--no-hsl", "--skip-connection", "--families", "course,object,user"] + TEMPORAL + LONG,
        user_rule="temporal"),
    "A5": scenario(
        "Ablation", "M0 bỏ skip MLP (chỉ nhánh graph)", "Nhánh graph tự đứng được đến đâu",
        ["--no-hsl", "--family-weights", "--families", "course,object,user"] + TEMPORAL + LONG,
        user_rule="temporal"),
    "C1": scenario(
        "Đối chứng", "A1 trên graph xáo trộn (giữ kích thước hyperedge, thành viên ngẫu nhiên)",
        "Lợi ích đến từ hàng xóm thật hay chỉ từ mô hình lớn hơn (so với A1)",
        MAIN + ["--families", "course,object", "--shuffle-graph", "7"] + LONG),
    "C2": scenario(
        "Đối chứng", "MLP: cùng encoder, chỉ self-loop", "Mốc không dùng graph",
        ["--no-hsl", "--families", "self_loop"] + LONG),
    "B-LR": scenario(
        "Baseline", "Logistic Regression trên X", "Mốc tuyến tính, không graph",
        ["--model", "logreg"], script=BASELINES),
    "B-GBDT": scenario(
        "Baseline", "GBDT (HistGradientBoosting) trên X", "Mốc cây quyết định, không graph",
        ["--model", "gbdt"], script=BASELINES),
    "B-HGNN": scenario(
        "Baseline", "HGNN gốc (Feng et al., 2019): Course, Object, User (any), không skip, không W",
        "Hypergraph cổ điển, cùng điều kiện với các baseline có tương lai",
        ["--no-hsl", "--families", "course,object,user"] + ANY + LONG, user_rule="any"),
    "B-HGNN-T": scenario(
        "Baseline", "HGNN gốc: Course, Object, User (temporal) + causal, không skip, không W",
        "Hypergraph cổ điển dưới cùng luật không tương lai với M0",
        ["--no-hsl", "--families", "course,object,user"] + TEMPORAL + LONG, user_rule="temporal"),
    **published("B-HyperGCN", "HyperGCN (Yadati et al., 2019) trên hypergraph Course, Object, User",
                "Hypergraph xấp xỉ bằng graph (mediator) thay cho HGNN", HYPERGCN,
                repo="HyperGCN", graph=True),
    **published("B-SIGNet", "SIG-Net (Kim et al., 2024): RGCN 3 cửa sổ + BiLSTM",
                "GNN trên graph học viên - object - khoá học", SIGNET, venv="signet", repo="SIG-Net"),
    **published("B-MSTGCN", "MST-GCN (2026): GCN dị thể + GRU đa thang thời gian",
                "GNN không gian - thời gian", MSTGCN, venv="mstgcn", repo="MST-GCN"),
    **published("B-CATFHN", "CA-TFHN (Liang et al., 2023): link prediction + graph bạn cùng lớp + TFHN",
                "Quan hệ bạn cùng lớp + chuỗi hành vi", CATFHN, venv="catfhn", repo="CA-TFHN"),
    **published("B-CFIN", "CFIN (Feng et al., 2019): attention theo ngữ cảnh học viên/khoá học",
                "Mô hình gốc của bộ dữ liệu XuetangX, không graph", CFIN),
}


def hypergraph_file(spec):
    if spec["bundle"]:
        return spec["bundle"]
    if spec["script"] != TRAIN:
        return None
    args = spec["args"]
    return args[args.index("--hypergraph") + 1] if "--hypergraph" in args else "hypergraph.npz"


# Problems that stop the scenario from running as described; empty when fine.
def problems(code):
    if code not in SCENARIOS:
        return [f"unknown scenario {code}; choose from {' '.join(SCENARIOS)}"]
    spec = SCENARIOS[code]
    found = []
    if spec["repo"] and not (ROOT / "baseline" / spec["repo"]).is_dir():
        found.append(f"{code}: baseline/{spec['repo']} is missing; run bash scripts/setup_baselines.sh")
    if spec["venv"] and not venv_python(spec["venv"]).exists():
        found.append(f"{code}: {venv_python(spec['venv'])} is missing; run bash scripts/setup_baselines.sh")
    file_name = hypergraph_file(spec)
    if file_name is None:
        return found
    path = PROCESSED / file_name
    if not path.exists():
        return found + [f"{code}: {path} is missing"]
    if spec["user_rule"] is None:
        return found
    with np.load(path) as bundle:
        rule = str(bundle["user_rule"]) if "user_rule" in bundle.files else None
    if rule != spec["user_rule"]:
        build = (f"python src/4_hypergraph.py --user-rule {spec['user_rule']} --hypergraph-file {file_name}"
                 " --reuse-neighbors <bundle with the same kNN>")
        return found + [f"{code}: {file_name} has User rule {rule!r}, needs {spec['user_rule']!r}. Build it: {build}"]
    return found


def venv_python(name):
    return ROOT / ".venvs" / name / "bin" / "python"


def main(command, codes):
    if command == "codes":
        print(" ".join(SCENARIOS))
    elif command == "list":
        for code, spec in SCENARIOS.items():
            print(f"{code:<7} {spec['group']:<10} {spec['description']}\n        -> {spec['question']}")
    elif command == "args" and len(codes) == 1 and codes[0] in SCENARIOS:
        print(SCENARIOS[codes[0]]["script"])
        print("\n".join(SCENARIOS[codes[0]]["args"]))
    elif command == "note" and len(codes) == 1 and codes[0] in SCENARIOS:
        spec = SCENARIOS[codes[0]]
        print(f"{codes[0]}: {spec['description']}")
    elif command == "python" and len(codes) == 1 and codes[0] in SCENARIOS:
        venv = SCENARIOS[codes[0]]["venv"]
        print(venv_python(venv) if venv else "")
    elif command == "bundle" and len(codes) == 1 and codes[0] in SCENARIOS:
        print(hypergraph_file(SCENARIOS[codes[0]]) or "")
    elif command == "cache" and codes:
        sys.exit(0 if any(SCENARIOS.get(code, {}).get("cache") for code in codes) else 1)
    elif command == "check" and codes:
        found = [problem for code in codes for problem in problems(code)]
        for problem in found:
            print(problem, file=sys.stderr)
        sys.exit(1 if found else 0)
    else:
        sys.exit("usage: python scripts/scenarios.py list | codes | args CODE | note CODE | python CODE"
                 " | bundle CODE | cache CODE [...] | check CODE [...]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "", sys.argv[2:])
