# Nơi duy nhất khai báo contract dữ liệu, feature schema và seed thực nghiệm.
from __future__ import annotations

from dataclasses import asdict, dataclass


SCHEMA_VERSION = "xuetangx-247-v1"
OBSERVATION_DAYS = 35
EARLY_OBSERVATION_DAYS = (7, 14, 21, 28, 35)

# tập dataset ban đầu gồm train và test
SOURCE_PARTITIONS = ("train", "test")
# split dataset thành train validation và test để train
EXPERIMENT_SPLITS = ("train", "validation", "test")
# thực nghiệm trên 5 seed và lấy giá trị mean +- std
EXPERIMENT_SEEDS = (1, 11, 111, 1111, 11111)

LOG_COLUMNS = (
    "enroll_id",
    "username",
    "course_id",
    "session_id",
    "action",
    "object",
    "time",
)
TRUTH_COLUMNS = ("enroll_id", "truth")
USER_COLUMNS = ("user_id", "gender", "education", "birth")
COURSE_COLUMNS = ("id", "course_id", "start", "end", "course_type", "category")
FULL_ACTIVITY_COLUMNS = ("course_id", "user_id", "session_id", "action", "time")

# Raw input contract. This is the only place that defines accepted filenames
# and columns.
SOURCE_SCHEMAS = {
    "train_log.csv": LOG_COLUMNS,
    "test_log.csv": LOG_COLUMNS,
    "train_truth.csv": TRUTH_COLUMNS,
    "test_truth.csv": TRUTH_COLUMNS,
    "user_info.csv": USER_COLUMNS,
    "course_info.csv": COURSE_COLUMNS,
}

ACTION_GROUPS = {
    "video": (
        "seek_video",
        "play_video",
        "pause_video",
        "stop_video",
        "load_video",
    ),
    "assignment": (
        "problem_get",
        "problem_check",
        "problem_save",
        "reset_problem",
        "problem_check_correct",
        "problem_check_incorrect",
    ),
    "forum": (
        "create_thread",
        "create_comment",
        "delete_thread",
        "delete_comment",
        "close_forum",
    ),
    "web_page": (
        "click_info",
        "click_courseware",
        "click_about",
        "click_forum",
        "click_progress",
        "close_courseware",
        "close_info",
    ),
}
ACTIONS = tuple(action for group in ACTION_GROUPS.values() for action in group)

GENDER_VALUES = ("female", "male")
EDUCATION_VALUES = (
    "Associate",
    "Bachelor's",
    "Doctorate",
    "High",
    "Master's",
    "Middle",
    "Primary",
)
COURSE_CATEGORY_VALUES = (
    "art",
    "biology",
    "business",
    "chemistry",
    "computer",
    "economics",
    "education",
    "electrical",
    "engineering",
    "foreign language",
    "history",
    "literature",
    "math",
    "medicine",
    "philosophy",
    "physics",
    "social science",
)
# dùng để làm ablation
# chỉ dùng feature
# dùng feature + user demographic
# dùng feature + course context
# dùng deature + user demographic + course context
FEATURE_SETS = (
    "behavior",
    "behavior_user",
    "behavior_course",
    "full",
)
# mặc định sử dụng feature, các mục còn lại đưa vào ablation
DEFAULT_FEATURE_SET = "behavior"


@dataclass(frozen=True)
# Mục đích: Gom các con số và quy ước cố định của thí nghiệm XuetangX-247.
# Đầu vào: Các giá trị được truyền khi khởi tạo DATASET_CONTRACT.
# Đầu ra: Object bất biến; có thể chuyển thành dictionary bằng to_dict().
# Lưu ý: Đây là contract để kiểm tra dữ liệu, không phải hyperparameter của model.
class DatasetContract:
    schema_version: str
    dataset: str
    enrollments: int
    users: int
    courses: int
    raw_events: int
    retained_events_35d: int
    observation_days: int
    experiment_seeds: tuple[int, ...]
    target_user_ratios: tuple[float, float, float]
    label_definition: str
    node_identity: str
    main_hyperedge_families: tuple[str, ...]

    # Mục đích: Chuyển contract thành dạng dễ ghi JSON hoặc in ra CLI.
    # Đầu vào: Chính object DatasetContract hiện tại.
    # Đầu ra: Dictionary chứa toàn bộ field của dataclass.
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DATASET_CONTRACT = DatasetContract(
    schema_version=SCHEMA_VERSION,
    dataset="XuetangX-247",
    enrollments=225_642,
    users=77_083,
    courses=247,
    raw_events=42_110_402,
    retained_events_35d=40_558_640,
    observation_days=OBSERVATION_DAYS,
    experiment_seeds=EXPERIMENT_SEEDS,
    target_user_ratios=(0.64, 0.16, 0.20),
    label_definition="truth=1 dropout; truth=0 non-dropout",
    node_identity="enroll_id",
    main_hyperedge_families=("course", "object", "behavioral"),
)


# Mục đích: Tạo danh sách feature hành vi theo đúng thứ tự cột của model.
# Đầu vào: Số ngày quan sát thuộc tập 7/14/21/28/35.
# Đầu ra: Tuple tên cột gồm daily counts, action counts và hai feature tổng hợp.
# Lưu ý: Thứ tự trả về phải khớp tuyệt đối với X_base.npy.
def base_feature_columns(observation_days: int = OBSERVATION_DAYS) -> tuple[str, ...]:
    if observation_days not in EARLY_OBSERVATION_DAYS:
        raise ValueError(
            f"observation_days must be one of {EARLY_OBSERVATION_DAYS}, "
            f"got {observation_days}"
        )
    daily = tuple(f"day_{day:02d}" for day in range(observation_days))
    action_counts = tuple(f"action_{action}" for action in ACTIONS)
    return daily + action_counts + ("session_count", "distinct_observed_objects")


# Mục đích: Chuẩn hóa một category thành phần tên cột dễ đọc.
# Đầu vào: Chuỗi category gốc.
# Đầu ra: Chuỗi chữ thường, bỏ dấu nháy và thay khoảng trắng bằng gạch dưới.
def _slug(value: str) -> str:
    return value.lower().replace("'", "").replace(" ", "_")


# Mục đích: Tạo schema cố định cho feature nhân khẩu học của user.
# Đầu vào: Không có; dùng các category đã khóa trong config.
# Đầu ra: Tuple 15 tên cột theo đúng thứ tự lưu trong X_context.npy.
# Lưu ý: Có cột missing và other để không làm mất mẫu lạ.
def user_feature_columns() -> tuple[str, ...]:
    gender = tuple(f"gender_{_slug(value)}" for value in GENDER_VALUES)
    education = tuple(
        f"education_{_slug(value)}" for value in EDUCATION_VALUES
    )
    return (
        *gender,
        "gender_missing",
        "gender_other",
        *education,
        "education_missing",
        "education_other",
        "age_at_course_start",
        "age_missing",
    )


# Mục đích: Tạo schema cố định cho feature bối cảnh course.
# Đầu vào: Không có; dùng COURSE_CATEGORY_VALUES đã khóa.
# Đầu ra: Tuple 21 tên cột theo đúng thứ tự lưu trong X_context.npy.
# Lưu ý: Hai cột cuối là duration và cờ duration bị thiếu.
def course_feature_columns() -> tuple[str, ...]:
    category = tuple(
        f"category_{_slug(value)}" for value in COURSE_CATEGORY_VALUES
    )
    return (
        *category,
        "category_missing",
        "category_other",
        "course_duration_days",
        "course_duration_missing",
    )


# Mục đích: Ghép schema user và course theo thứ tự của context array.
# Đầu vào: Không có.
# Đầu ra: Tuple 36 tên feature context.
def context_feature_columns() -> tuple[str, ...]:
    return user_feature_columns() + course_feature_columns()


# Mục đích: Chọn schema đầu vào tương ứng với một cấu hình ablation feature.
# Đầu vào: Tên feature_set và số ngày quan sát.
# Đầu ra: Tuple tên cột có 60, 75, 81 hoặc 96 phần tử.
# Lưu ý: Ném ValueError nếu feature_set không thuộc FEATURE_SETS.
def feature_columns(
    feature_set: str, observation_days: int = OBSERVATION_DAYS
) -> tuple[str, ...]:
    behavior = base_feature_columns(observation_days)
    if feature_set == "behavior":
        return behavior
    if feature_set == "behavior_user":
        return behavior + user_feature_columns()
    if feature_set == "behavior_course":
        return behavior + course_feature_columns()
    if feature_set == "full":
        return behavior + context_feature_columns()
    raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")


# Mục đích: Phát hiện lỗi cấu hình ngay khi module được import.
# Đầu vào: Không có; đọc các hằng số phía trên.
# Đầu ra: Không trả dữ liệu nếu contract hợp lệ.
# Lưu ý: Ném RuntimeError khi số action, số feature, tỷ lệ split hoặc seed bị sai.
def _validate_contract() -> None:
    if len(ACTIONS) != 23 or len(set(ACTIONS)) != len(ACTIONS):
        raise RuntimeError("Action vocabulary must contain 23 unique actions.")
    if len(base_feature_columns()) != 60:
        raise RuntimeError("The 35-day X_base schema must contain 60 features.")
    if len(user_feature_columns()) != 15:
        raise RuntimeError("The user-demographic schema must contain 15 features.")
    if len(course_feature_columns()) != 21:
        raise RuntimeError("The course-context schema must contain 21 features.")
    expected_dimensions = {
        "behavior": 60,
        "behavior_user": 75,
        "behavior_course": 81,
        "full": 96,
    }
    for feature_set, dimension in expected_dimensions.items():
        columns = feature_columns(feature_set)
        if len(columns) != dimension or len(set(columns)) != dimension:
            raise RuntimeError(
                f"Invalid {feature_set} feature schema: expected {dimension} unique columns."
            )
    if abs(sum(DATASET_CONTRACT.target_user_ratios) - 1.0) > 1e-12:
        raise RuntimeError("Target split ratios must sum to one.")
    if (
        len(EXPERIMENT_SEEDS) != 5
        or len(set(EXPERIMENT_SEEDS)) != 5
        or any(seed <= 0 for seed in EXPERIMENT_SEEDS)
    ):
        raise RuntimeError("The experiment must use five unique seeds.")


_validate_contract()
