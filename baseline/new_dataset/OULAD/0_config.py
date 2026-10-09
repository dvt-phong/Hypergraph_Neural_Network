# 0. Setting hyperparameter and config paramater (OULAD)
# - OULAD https://doi.org/10.1038/sdata.2017.171
# - Wu et al. (2026) code: run_oulad_revision_pipeline.py (Zenodo)

from pathlib import Path


# Project paths and experiment seeds.
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
SPLIT_SEED = 1
TEST_RATIO = 0.20     # last 20% of the shuffled enrollments -> test
TRAIN_RATIO = 0.80    # the other 80% -> train / validation 80/20, as XuetangX
OBSERVATION_DAYS = 35
DROPOUT_RESULT = "Withdrawn"   # label = 1[final_result = Withdrawn] (Wu et al.)
MISSING_VALUES = ("", "?")


# Used by 3_features.py
# Activity types with clicks in days 0–34 (Wu et al. keep the types seen in the window; folder and
# repeatactivity have none), in alphabetical order as Wu's unstack.
ACTIVITY_TYPES = (
    "dataplus", "dualpane", "externalquiz", "forumng", "glossary", "homepage", "htmlactivity",
    "oucollaborate", "oucontent", "ouelluminate", "ouwiki", "page", "questionnaire", "quiz",
    "resource", "sharedsubpage", "subpage", "url",
)
GENDERS = ("F", "M")
AGE_BANDS = ("0-35", "35-55", "55<=")
REGIONS = (
    "East Anglian Region", "East Midlands Region", "Ireland", "London Region", "North Region",
    "North Western Region", "Scotland", "South East Region", "South Region", "South West Region",
    "Wales", "West Midlands Region", "Yorkshire Region",
)
EDUCATIONS = (
    "A Level or Equivalent", "HE Qualification", "Lower Than A Level", "No Formal quals",
    "Post Graduate Qualification",
)
IMD_BANDS = (
    "0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
)
MODULES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")
PRESENTATIONS = ("2013B", "2013J", "2014B", "2014J")

# X = [behavior | user | course]
# behavior: 35 daily clicks | total clicks | 18 activity-type clicks | 6 assessment columns (Wu)
DAY_FEATURE_COUNT = OBSERVATION_DAYS
TOTAL_CLICKS_INDEX = DAY_FEATURE_COUNT                       # clicks in days 0–34
ACTIVITY_FEATURE_START = TOTAL_CLICKS_INDEX + 1
ACTIVITY_FEATURE_COUNT = len(ACTIVITY_TYPES)
N_ASSESS_INDEX = ACTIVITY_FEATURE_START + ACTIVITY_FEATURE_COUNT
SUBMITTED_ANY_INDEX = N_ASSESS_INDEX + 1
AVG_SCORE_INDEX = SUBMITTED_ANY_INDEX + 1
WEIGHTED_SCORE_INDEX = AVG_SCORE_INDEX + 1
MEAN_LATENESS_INDEX = WEIGHTED_SCORE_INDEX + 1
MAX_LATENESS_INDEX = MEAN_LATENESS_INDEX + 1
BEHAVIOR_FEATURE_COUNT = MAX_LATENESS_INDEX + 1

# user: one-hot gender, age_band, region, highest_education, imd_band (+ missing) | studied_credits,
# num_of_prev_attempts, date_registration. disability stays out of X (protected attribute, Wu et al.).
# gender and age_band have no missing values, so no missing column (XuetangX gender has one).
GENDER_FEATURE_START = 0
AGE_FEATURE_START = GENDER_FEATURE_START + len(GENDERS)
REGION_FEATURE_START = AGE_FEATURE_START + len(AGE_BANDS)
EDUCATION_FEATURE_START = REGION_FEATURE_START + len(REGIONS)
IMD_FEATURE_START = EDUCATION_FEATURE_START + len(EDUCATIONS)
IMD_FEATURE_COUNT = len(IMD_BANDS) + 1                      # imd_band is the only field with missing values
CREDITS_FEATURE_INDEX = IMD_FEATURE_START + IMD_FEATURE_COUNT
PREVIOUS_ATTEMPTS_FEATURE_INDEX = CREDITS_FEATURE_INDEX + 1
REGISTRATION_FEATURE_INDEX = PREVIOUS_ATTEMPTS_FEATURE_INDEX + 1
USER_FEATURE_COUNT = REGISTRATION_FEATURE_INDEX + 1

# course: one-hot code_module, code_presentation
MODULE_FEATURE_START = 0
PRESENTATION_FEATURE_START = MODULE_FEATURE_START + len(MODULES)
COURSE_FEATURE_COUNT = PRESENTATION_FEATURE_START + len(PRESENTATIONS)

USER_FEATURE_START = BEHAVIOR_FEATURE_COUNT
COURSE_FEATURE_START = USER_FEATURE_START + USER_FEATURE_COUNT
TOTAL_FEATURE_COUNT = COURSE_FEATURE_START + COURSE_FEATURE_COUNT

# Scaling with train-only statistics:
#   LOG_Z_COLUMNS: counts, x = (log(1 + c) − μ_train) / σ_train
#   Z_COLUMNS:     other numbers, x = (v − μ_train) / σ_train
# submitted_any and the one-hot columns stay 0/1.
LOG_Z_COLUMNS = tuple(range(0, SUBMITTED_ANY_INDEX))
Z_COLUMNS = (
    AVG_SCORE_INDEX, WEIGHTED_SCORE_INDEX, MEAN_LATENESS_INDEX, MAX_LATENESS_INDEX,
    USER_FEATURE_START + CREDITS_FEATURE_INDEX,
    USER_FEATURE_START + PREVIOUS_ATTEMPTS_FEATURE_INDEX,
    USER_FEATURE_START + REGISTRATION_FEATURE_INDEX,
)


# Used by 4_hypergraph.py and 6_hgnn.py.
# Every click is on a VLE site (id_site), so every activity type is an object.
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
