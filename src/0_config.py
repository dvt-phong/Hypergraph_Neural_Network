# Central configuration shared by all pipeline steps.

from pathlib import Path


# Shared project paths and experiment seeds.
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "xuetangx"
PROCESSED = ROOT / "data" / "processed" / "simple"
RUNS = ROOT / "outputs" / "runs"
REPORTS = ROOT / "outputs" / "reports"
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
DOWNLOAD_CLI_DESCRIPTION = "Download the raw XuetangX dataset."


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
ACTIONS = []
for action_group in ACTION_GROUPS.values():
    for action_name in action_group:
        ACTIONS.append(action_name)
ACTIONS = tuple(ACTIONS)
TRUTH_FILES = ("train_truth.csv", "test_truth.csv")
LOG_FILES = ("train_log.csv", "test_log.csv")
PREPROCESS_CLI_DESCRIPTION = "Preprocess the XuetangX dataset."


# Used by 3_features.py.
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

DAY_FEATURE_COUNT = OBSERVATION_DAYS
ACTION_FEATURE_START = DAY_FEATURE_COUNT
ACTION_FEATURE_COUNT = len(ACTIONS)
BEHAVIOR_FEATURE_COUNT = ACTION_FEATURE_START + ACTION_FEATURE_COUNT

GENDER_FEATURE_START = 0
GENDER_FEATURE_COUNT = len(GENDERS) + 2
EDUCATION_FEATURE_START = GENDER_FEATURE_START + GENDER_FEATURE_COUNT
EDUCATION_FEATURE_COUNT = len(EDUCATIONS) + 2
AGE_FEATURE_INDEX = EDUCATION_FEATURE_START + EDUCATION_FEATURE_COUNT
AGE_MISSING_FEATURE_INDEX = AGE_FEATURE_INDEX + 1
USER_FEATURE_COUNT = AGE_MISSING_FEATURE_INDEX + 1

CATEGORY_FEATURE_START = 0
CATEGORY_FEATURE_COUNT = len(CATEGORIES) + 2
COURSE_FEATURE_COUNT = CATEGORY_FEATURE_COUNT

USER_FEATURE_START = BEHAVIOR_FEATURE_COUNT
COURSE_FEATURE_START = USER_FEATURE_START + USER_FEATURE_COUNT
TOTAL_FEATURE_COUNT = COURSE_FEATURE_START + COURSE_FEATURE_COUNT
BEHAVIOR_FEATURE_SLICE = slice(0, BEHAVIOR_FEATURE_COUNT)

OBJECT_ACTIONS = {}
for object_family, object_actions in ACTION_GROUPS.items():
    if object_family != "web_page":
        for object_action in object_actions:
            OBJECT_ACTIONS[object_action] = object_family
FEATURE_CLI_DESCRIPTION = "Build train, validation, and test node features."


# Used by 4_hypergraph.py and 6_hsl.py. 
# Every node also gets one self-loop
# hyperedge that HSL never removes (Cai et al., 2022, Eq. 9).
EDGE_FAMILIES = ("course", "object", "behavioral", "self_loop")
HYPERGRAPH_CLI_DESCRIPTION = "Build Course, Object, and Behavioral hypergraphs."


# Used by 5_model.py and 6_hsl.py. Memberships are processed in chunks of this
# size so that the full XuetangX graph fits in GPU memory.
MEMBERSHIP_CHUNK_SIZE = 1_000_000


# Used by 8_train.py.
TRAIN_CLI_DESCRIPTION = "Train HGSL, select on validation, and evaluate on test."
