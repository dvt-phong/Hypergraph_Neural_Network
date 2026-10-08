# 0. Setting hyperparameter and config paramater

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
        "click_progress", "close_courseware",
    ),
}
# All 22 actions in one tuple, group by group.
action_list = []
for group_name in ACTION_GROUPS:
    for action in ACTION_GROUPS[group_name]:
        action_list.append(action)
ACTIONS = tuple(action_list)
TRUTH_FILES = ("train_truth.csv", "test_truth.csv")
LOG_FILES = ("train_log.csv", "test_log.csv")


# Used by 3_features.py
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
AGE_MIN = 10
AGE_MAX = 70
AGE_MISSING = 0

# X = [behavior | user (gender, education, age) | course (category)]
# 35 daily event counts | total events, distinct objects, 22 action counts
DAY_FEATURE_COUNT = OBSERVATION_DAYS
TOTAL_EVENTS_INDEX = DAY_FEATURE_COUNT                 # events in days 0–34
OBJECT_COUNT_INDEX = TOTAL_EVENTS_INDEX + 1            # distinct objects used in days 0–34
ACTION_FEATURE_START = OBJECT_COUNT_INDEX + 1
ACTION_FEATURE_COUNT = len(ACTIONS)
BEHAVIOR_FEATURE_COUNT = ACTION_FEATURE_START + ACTION_FEATURE_COUNT

GENDER_FEATURE_START = 0
GENDER_FEATURE_COUNT = len(GENDERS) + 1
EDUCATION_FEATURE_START = GENDER_FEATURE_START + GENDER_FEATURE_COUNT
EDUCATION_FEATURE_COUNT = len(EDUCATIONS) + 1
AGE_FEATURE_INDEX = EDUCATION_FEATURE_START + EDUCATION_FEATURE_COUNT  # user[:, 11] -> final X[:, 70], z-score
USER_FEATURE_COUNT = AGE_FEATURE_INDEX + 1

CATEGORY_FEATURE_START = 0
CATEGORY_FEATURE_COUNT = len(CATEGORIES) + 1
COURSE_FEATURE_COUNT = CATEGORY_FEATURE_COUNT

USER_FEATURE_START = BEHAVIOR_FEATURE_COUNT
COURSE_FEATURE_START = USER_FEATURE_START + USER_FEATURE_COUNT
TOTAL_FEATURE_COUNT = COURSE_FEATURE_START + COURSE_FEATURE_COUNT


# Used by 4_hypergraph.py and 6_hgnn.py.
OBJECT_ACTIONS = {}
for group_name in ACTION_GROUPS:
    if group_name == "web_page":
        continue
    for action in ACTION_GROUPS[group_name]:
        OBJECT_ACTIONS[action] = group_name
# Hyperedge families. Every node also gets one self-loop hyperedge when the graph is loaded.
EDGE_FAMILIES = ("course", "object", "user", "self_loop")


# Used by 9_train.py and 10_summary.py
def scenario(description, families=EDGE_FAMILIES, features="full", hgnn_layers=2, use_mlp=True,
             learn_w=True):
    return {"description": description, "families": tuple(families), "features": features,
            "hgnn_layers": hgnn_layers, "use_mlp": use_mlp, "learn_w": learn_w}


SCENARIOS = {
    "M":  scenario("Mô hình chính: Course + Object + User + self-loop, full, HGNN 2 layer, có MLP"),
    "A1": scenario("M bỏ Course", families=("object", "user", "self_loop")),
    "A2": scenario("M bỏ Object", families=("course", "user", "self_loop")),
    "A3": scenario("M bỏ User", families=("course", "object", "self_loop")),
    "A4": scenario("M bỏ self-loop", families=("course", "object", "user")),
    "W1": scenario("M không học W (W = I cố định, như HGNN gốc)", learn_w=False),
    "F1": scenario("M chỉ dùng feature hành vi", features="feature"),
    "F2": scenario("M dùng feature hành vi + người học", features="feature+user"),
    "F3": scenario("M dùng feature hành vi + khóa học", features="feature+course"),
    "L1": scenario("M với HGNN 1 layer", hgnn_layers=1),
    "H":  scenario("M bỏ nhánh MLP, chỉ còn hypergraph", use_mlp=False),
    # Optional, not part of "all": no neighbours at all (G = I, the HGNN branch becomes an MLP).
    "X1": scenario("Tùy chọn: chỉ self-loop, không có hàng xóm", families=("self_loop",)),
}

SCENARIOS_ALL = ("M", "A1", "A2", "A3", "A4", "W1", "F1", "F2", "F3", "L1", "H")


# Used by 9_train.py.
THRESHOLD = 0.5  # p >= 0.5 -> predicted dropout
TRAIN = {
    "hidden_dim": 128,        
    "dropout": 0.5,            
    "learning_rate": 1e-3,     
    "weight_decay": 5e-4,   
    "family_weight_lr": 0.05,  
    "epochs": 1000,         
    "eval_every": 5,            
    "patience": 40,           
}
