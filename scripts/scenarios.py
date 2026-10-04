# The experiment scenarios of the paper, in one place. Every scenario runs
# src/6_train.py; the scenario code is also the run tag (--tag).
#
#   python scripts/scenarios.py list            code, group, description, question
#   python scripts/scenarios.py codes           every code, in table order
#   python scripts/scenarios.py args M0         arguments of 6_train.py, one per line
#   python scripts/scenarios.py note M0         one-line note for the workbook
#   python scripts/scenarios.py bundle M0       hypergraph bundle the scenario reads
#   python scripts/scenarios.py check M0 [...]  the bundle exists and holds the User
#                                               rule the scenario needs
#
# scripts/run_scenarios.sh runs them. Settings not given here are the defaults
# of DEFAULT_SETTINGS in src/6_train.py (hidden 128, dropout 0.5, lr 1e-3,
# weight decay 5e-4, family weight lr 0.05, validation every 5 epochs,
# checkpoint on validation AUPRC, t* on the full validation split).
#
# User hyperedges come from two bundles in data/processed/simple:
#   hypergraph.npz           rule "any": all courses of the learner, also later
#                            ones (uses future information)
#   hypergraph_temporal.npz  rule "temporal": only courses that started no later;
#                            with --causal nothing reaches a node from after its
#                            35-day window (scripts/check_leakage.py)

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "simple"

LONG = ["--epochs", "1000", "--patience", "60"]
MAIN = ["--skip-connection", "--family-weights"]  # HGNN + skip MLP(X) + W per family
ANY = ["--hypergraph", "hypergraph.npz"]
TEMPORAL = ["--hypergraph", "hypergraph_temporal.npz", "--causal"]


def scenario(group, description, question, args, user_rule=None):
    return {"group": group, "description": description, "question": question,
            "args": args, "user_rule": user_rule}


SCENARIOS = {
    "M-any": scenario(
        "Chính", "HGNN + skip MLP + W; Course, Object, User (any), self-loop",
        "Cùng điều kiện với baseline (User có cả khoá học tương lai); tái hiện 0,8749 (u1-hgnn)",
        MAIN + ["--families", "course,object,user"] + ANY + LONG, "any"),
    "M0": scenario(
        "Chính", "Như M-any, User temporal + causal",
        "Kết quả chính không rò rỉ thông tin tương lai",
        MAIN + ["--families", "course,object,user"] + TEMPORAL + LONG, "temporal"),
    "A1": scenario(
        "Ablation", "M0 bỏ User (Course, Object)", "User đóng góp bao nhiêu khi không rò rỉ",
        MAIN + ["--families", "course,object"] + LONG),
    "A2": scenario(
        "Ablation", "M0 bỏ Object (Course, User temporal)", "Vai trò của Object",
        MAIN + ["--families", "course,user"] + TEMPORAL + LONG, "temporal"),
    "A3": scenario(
        "Ablation", "M0 bỏ Course (Object, User temporal)", "Vai trò của Course",
        MAIN + ["--families", "object,user"] + TEMPORAL + LONG, "temporal"),
    "A4": scenario(
        "Ablation", "M0 bỏ W (mọi loại hyperedge cùng trọng số)", "W theo loại hyperedge có cần không",
        ["--skip-connection", "--families", "course,object,user"] + TEMPORAL + LONG, "temporal"),
    "A5": scenario(
        "Ablation", "M0 bỏ skip MLP (chỉ nhánh graph)", "Nhánh graph tự đứng được đến đâu",
        ["--family-weights", "--families", "course,object,user"] + TEMPORAL + LONG, "temporal"),
    "C1": scenario(
        "Đối chứng", "A1 trên graph xáo trộn (giữ kích thước hyperedge, thành viên ngẫu nhiên)",
        "Lợi ích đến từ hàng xóm thật hay chỉ từ mô hình lớn hơn (so với A1)",
        MAIN + ["--families", "course,object", "--shuffle-graph", "7"] + LONG),
    "C2": scenario(
        "Đối chứng", "MLP: cùng encoder, chỉ self-loop", "Mốc không dùng graph",
        ["--families", "self_loop"] + LONG),
    "B-HGNN": scenario(
        "Baseline", "HGNN gốc (Feng et al., 2019): Course, Object, User (any), không skip, không W",
        "Hypergraph cổ điển, cùng điều kiện với các baseline có tương lai",
        ["--families", "course,object,user"] + ANY + LONG, "any"),
    "B-HGNN-T": scenario(
        "Baseline", "HGNN gốc: Course, Object, User (temporal) + causal, không skip, không W",
        "Hypergraph cổ điển dưới cùng luật không tương lai với M0",
        ["--families", "course,object,user"] + TEMPORAL + LONG, "temporal"),
}


# Hypergraph bundle a scenario reads (default hypergraph.npz).
def hypergraph_file(spec):
    args = spec["args"]
    return args[args.index("--hypergraph") + 1] if "--hypergraph" in args else "hypergraph.npz"


# Problems that stop the scenario from running as described; empty when fine.
def problems(code):
    if code not in SCENARIOS:
        return [f"unknown scenario {code}; choose from {' '.join(SCENARIOS)}"]
    spec = SCENARIOS[code]
    file_name = hypergraph_file(spec)
    path = PROCESSED / file_name
    if not path.exists():
        return [f"{code}: {path} is missing"]
    if spec["user_rule"] is None:
        return []
    with np.load(path) as bundle:
        rule = str(bundle["user_rule"]) if "user_rule" in bundle.files else None
    if rule != spec["user_rule"]:
        build = (f"python src/4_hypergraph.py --user-rule {spec['user_rule']} --hypergraph-file {file_name}"
                 " --reuse-neighbors <bundle with the same kNN>")
        return [f"{code}: {file_name} has User rule {rule!r}, needs {spec['user_rule']!r}. Build it: {build}"]
    return []


def main(command, codes):
    one = len(codes) == 1 and codes[0] in SCENARIOS
    if command == "codes":
        print(" ".join(SCENARIOS))
    elif command == "list":
        for code, spec in SCENARIOS.items():
            print(f"{code:<9} {spec['group']:<10} {spec['description']}\n          -> {spec['question']}")
    elif command == "args" and one:
        print("\n".join(SCENARIOS[codes[0]]["args"]))
    elif command == "note" and one:
        print(f"{codes[0]}: {SCENARIOS[codes[0]]['description']}")
    elif command == "bundle" and one:
        print(hypergraph_file(SCENARIOS[codes[0]]))
    elif command == "check" and codes:
        found = [problem for code in codes for problem in problems(code)]
        for problem in found:
            print(problem, file=sys.stderr)
        sys.exit(1 if found else 0)
    else:
        sys.exit("usage: python scripts/scenarios.py list | codes | args CODE | note CODE"
                 " | bundle CODE | check CODE [...]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "", sys.argv[2:])
