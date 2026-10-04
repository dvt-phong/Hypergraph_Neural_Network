# 0. Settings shared by every step: paths, seeds, feature layout, hyperedge families,
#    training hyperparameters.

from pathlib import Path


# Project paths and experiment seeds.
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "xuetangx"
PROCESSED = ROOT / "data" / "processed" / "simple"
OUTPUTS = ROOT / "outputs"
RESULTS_CSV = OUTPUTS / "results.csv"
SEEDS = (1, 11, 111, 1111, 11111)


# Used by 1_download.py.
DOWNLOAD_FILES = {
    "prediction_data.tar.gz":
        "https://lfs.aminer.cn/misc/moocdata/data/prediction_data.tar.gz",
    "user_info.csv":
        "https://lfs.aminer.cn/misc/moocdata/data/user_info.csv",
    "course_info.csv":
        "https://lfs.aminer.cn/misc/moocdata/data/course_info.csv",
}


# Used by 2_preprocess.py.
SPLITS = ("train", "validation", "test")
SPLIT_SEED = 1
TRAIN_RATIO = 0.80
OBSERVATION_DAYS = 35
ACTION_GROUPS = {
    "video": (
        "seek_video", "play_video", "pause_video", "stop_video", "load_video"
    ),
    "assignment": (
        "problem_get", "problem_check", "problem_save", "reset_problem",
        "problem_check_correct", "problem_check_incorrect",
    ),
    "forum": (
        "create_thread", "create_comment", "delete_thread", "delete_comment",
        "close_forum",
    ),
    "web_page": (
        "click_info", "click_courseware", "click_about", "click_forum",
        "click_progress", "close_courseware", "close_info",
    ),
}
ACTIONS = tuple(action for actions in ACTION_GROUPS.values() for action in actions)
TRUTH_FILES = ("train_truth.csv", "test_truth.csv")
LOG_FILES = ("train_log.csv", "test_log.csv")


# Used by 3_features.py: vocabularies and the column layout of X.
GENDERS = ("female", "male")
EDUCATIONS = (
    "Associate", "Bachelor's", "Doctorate", "High", "Master's", "Middle", "Primary"
)
CATEGORIES = (
    "art", "biology", "business", "chemistry", "computer", "economics",
    "education", "electrical", "engineering", "foreign language", "history",
    "literature", "math", "medicine", "philosophy", "physics", "social science",
)
MISSING_VALUES = ("", "na", "n/a", "none", "null", "no data", "-")

# X = [behavior (35 days + 23 actions) | user (gender, education) | course (category)]
# Age is left out: birth is missing for 71.5% of the enrollments.
DAY_FEATURE_COUNT = OBSERVATION_DAYS
ACTION_FEATURE_START = DAY_FEATURE_COUNT
ACTION_FEATURE_COUNT = len(ACTIONS)
BEHAVIOR_FEATURE_COUNT = ACTION_FEATURE_START + ACTION_FEATURE_COUNT

# Each one-hot block ends with a "missing" and an "other" column.
GENDER_FEATURE_START = 0
GENDER_FEATURE_COUNT = len(GENDERS) + 2
EDUCATION_FEATURE_START = GENDER_FEATURE_START + GENDER_FEATURE_COUNT
EDUCATION_FEATURE_COUNT = len(EDUCATIONS) + 2
USER_FEATURE_COUNT = EDUCATION_FEATURE_START + EDUCATION_FEATURE_COUNT

CATEGORY_FEATURE_START = 0
CATEGORY_FEATURE_COUNT = len(CATEGORIES) + 2
COURSE_FEATURE_COUNT = CATEGORY_FEATURE_COUNT

USER_FEATURE_START = BEHAVIOR_FEATURE_COUNT
COURSE_FEATURE_START = USER_FEATURE_START + USER_FEATURE_COUNT
TOTAL_FEATURE_COUNT = COURSE_FEATURE_START + COURSE_FEATURE_COUNT


# Used by 4_hypergraph.py and 6_hgnn.py.
# Actions on an object (video, problem, forum post) -> object family; web pages have no object.
OBJECT_ACTIONS = {
    action: family
    for family, actions in ACTION_GROUPS.items() if family != "web_page"
    for action in actions
}
# Hyperedge families. Every node also gets one self-loop hyperedge when the graph is loaded.
EDGE_FAMILIES = ("course", "object", "user", "self_loop")


# Used by 9_train.py and 10_summary.py: the experiment scenarios (docs/KICH_BAN_THUC_NGHIEM.md).
# Every scenario changes ONE factor of the main model M:
#   families     hyperedge families the graph keeps ("self_loop" is one of them)
#   features     columns of X (3_features.feature_columns):
#                "feature" = behavior, "feature+user", "feature+course", "full" = all 90
#   hgnn_layers  1 or 2 HGNN layers in the graph branch (the MLP branch is always 2 layers)
#   use_mlp      keep the MLP branch next to the HGNN branch
#   learn_w      learn one weight per hyperedge family (W); False = W = I fixed, as in the
#                original HGNN code
#   description  one line for the results table
def scenario(description, *, families=EDGE_FAMILIES, features="full", hgnn_layers=2, use_mlp=True,
             learn_w=True):
    return {"description": description, "families": tuple(families), "features": features,
            "hgnn_layers": hgnn_layers, "use_mlp": use_mlp, "learn_w": learn_w}


SCENARIOS = {
    "M":  scenario("Mô hình chính: Course + Object + User + self-loop, full, HGNN 2 layer, có MLP"),
    "A1": scenario("M bỏ Course", families=("object", "user", "self_loop")),
    "A2": scenario("M bỏ Object", families=("course", "user", "self_loop")),
    "A3": scenario("M bỏ User", families=("course", "object", "self_loop")),
    "A4": scenario("M bỏ self-loop", families=("course", "object", "user")),
    "F1": scenario("M chỉ dùng feature hành vi", features="feature"),
    "F2": scenario("M dùng feature hành vi + người học", features="feature+user"),
    "F3": scenario("M dùng feature hành vi + khóa học", features="feature+course"),
    "L1": scenario("M với HGNN 1 layer", hgnn_layers=1),
    "B1": scenario("M bỏ nhánh MLP (chỉ còn HGNN)", use_mlp=False),
    "W1": scenario("M không học W (W = I cố định, như HGNN gốc)", learn_w=False),
    # Optional, not part of "all": no neighbours at all (G = I, the HGNN branch becomes an MLP).
    "X1": scenario("Tùy chọn: chỉ self-loop, không có hàng xóm", families=("self_loop",)),
}
# What `--scenario all` runs, in this order (X1 is left out on purpose).
SCENARIOS_ALL = ("M", "A1", "A2", "A3", "A4", "F1", "F2", "F3", "L1", "B1", "W1")


# Used by 9_train.py.
THRESHOLD = 0.5  # p >= 0.5 -> predicted dropout
TRAIN = {
    "hidden_dim": 128,          # as in HGNN (Feng et al., 2019)
    "dropout": 0.5,             # as in HGNN (Feng et al., 2019)
    "learning_rate": 1e-3,
    "weight_decay": 5e-4,       # L2 on every weight except the family weights
    "family_weight_lr": 0.05,   # own lr for the family weights W, no weight decay
    "epochs": 1000,             # maximum number of epochs
    "eval_every": 5,            # validation AUC every N epochs
    "patience": 40,             # stop after N validations without a better AUC
    "eval_batch_size": 4096,    # validation/test targets per batch
}
