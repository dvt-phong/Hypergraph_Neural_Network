# 0. Setting hyperparameter and config paramater (OULAD)
# - OULAD https://doi.org/10.1038/sdata.2017.171
# - Wu et al. (2026) code: run_oulad_revision_pipeline.py (Zenodo)

from pathlib import Path


# Project paths and experiment seeds.
# Seed s splits every enrollment into train / validation / test and initialises the model.
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "oulad"
PROCESSED = ROOT / "data" / "processed" / "oulad"
OUTPUTS = ROOT / "outputs" / "oulad"
RESULTS_CSV = OUTPUTS / "results.csv"
SEEDS = (1, 11, 111, 1111, 11111)


# Used by 1_download.py.
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
ZIP_FILE = "oulad.zip"
RAW_FILES = (
    "courses.csv", "assessments.csv", "vle.csv", "studentInfo.csv",
    "studentRegistration.csv", "studentAssessment.csv", "studentVle.csv",
)


# Used by 2_preprocess.py.
SPLITS = ("train", "validation", "test")
# Every enrollment -> 60% train, 10% validation, 30% test, stratified by label, with the seed
# (as XuetangX, src/0_config.py).
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.30
OBSERVATION_DAYS = 35
DROPOUT_RESULT = "Withdrawn"   # label = 1[final_result = Withdrawn] (Wu et al.)
MISSING_VALUES = ("", "?")


# Used by 3_features.py
# Activity types with clicks in days 0–34 (folder and repeatactivity have none), in alphabetical order.
ACTIVITY_TYPES = (
    "dataplus", "dualpane", "externalquiz", "forumng", "glossary", "homepage", "htmlactivity",
    "oucollaborate", "oucontent", "ouelluminate", "ouwiki", "page", "questionnaire", "quiz",
    "resource", "sharedsubpage", "subpage", "url",
)
# X55 = [35 daily clicks | total clicks | distinct sites | 18 activity-type clicks]
# (the OULAD version of X59 of XuetangX: log activity only)
X_NAME = "x55"
DAY_FEATURE_COUNT = OBSERVATION_DAYS
TOTAL_CLICKS_INDEX = DAY_FEATURE_COUNT                 # clicks in days 0–34
SITE_COUNT_INDEX = TOTAL_CLICKS_INDEX + 1              # distinct VLE sites clicked in days 0–34
ACTIVITY_FEATURE_START = SITE_COUNT_INDEX + 1
ACTIVITY_FEATURE_COUNT = len(ACTIVITY_TYPES)
FEATURE_COUNT = ACTIVITY_FEATURE_START + ACTIVITY_FEATURE_COUNT   # 55


# Used by 4_hypergraph.py and 6_hgnn.py.
# Every click is on a VLE site (id_site), so every site is an object.
# Hyperedge families. Every node also gets one self-loop hyperedge when the graph is loaded.
EDGE_FAMILIES = ("course", "object", "user", "self_loop")


# Used by 9_train.py and 10_summary.py
def scenario(description, families=EDGE_FAMILIES, hgnn_layers=2, use_mlp=True, learn_w=True):
    return {"description": description, "families": tuple(families),
            "hgnn_layers": hgnn_layers, "use_mlp": use_mlp, "learn_w": learn_w}


SCENARIOS = {
    "M":  scenario("Mô hình chính: Course + Object + User + self-loop, HGNN 2 layer, có MLP"),
    "A1": scenario("M bỏ Course", families=("object", "user", "self_loop")),
    "A2": scenario("M bỏ Object", families=("course", "user", "self_loop")),
    "A3": scenario("M bỏ User", families=("course", "object", "self_loop")),
    "A4": scenario("M bỏ self-loop", families=("course", "object", "user")),
    "W1": scenario("M không học W (W = I cố định, như HGNN gốc)", learn_w=False),
    "L1": scenario("M với HGNN 1 layer", hgnn_layers=1),
    "H":  scenario("M bỏ nhánh MLP, chỉ còn hypergraph", use_mlp=False),
    # Optional, not part of "all": no neighbours at all (G = I, the HGNN branch becomes an MLP).
    "X1": scenario("Tùy chọn: chỉ self-loop, không có hàng xóm", families=("self_loop",)),
}

SCENARIOS_ALL = ("M", "A1", "A2", "A3", "A4", "W1", "L1", "H")


# Used by 9_train.py (the same settings as XuetangX, src/0_config.py).
THRESHOLD = 0.5  # p >= 0.5 -> predicted dropout
TRAIN = {
    "hidden_dim": 128,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "family_weight_lr": 0.05,
    "epochs": 1000,
    "eval_every": 1,
    "patience": 200,
}
