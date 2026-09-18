"""Single source of truth for the XuetangX-247 data contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass


SCHEMA_VERSION = "xuetangx-247-v1"
OBSERVATION_DAYS = 35
EARLY_OBSERVATION_DAYS = (7, 14, 21, 28, 35)

SOURCE_PARTITIONS = ("train", "test")
EXPERIMENT_SPLITS = ("train", "validation", "test")
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


@dataclass(frozen=True)
class DatasetContract:
    """Expected scope of the supervised experiment."""

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


def base_feature_columns(observation_days: int = OBSERVATION_DAYS) -> tuple[str, ...]:
    """Return the ordered main feature schema for an observation window."""

    if observation_days not in EARLY_OBSERVATION_DAYS:
        raise ValueError(
            f"observation_days must be one of {EARLY_OBSERVATION_DAYS}, "
            f"got {observation_days}"
        )
    daily = tuple(f"day_{day:02d}" for day in range(observation_days))
    action_counts = tuple(f"action_{action}" for action in ACTIONS)
    return daily + action_counts + ("session_count", "distinct_observed_objects")


def _validate_contract() -> None:
    if len(ACTIONS) != 23 or len(set(ACTIONS)) != len(ACTIONS):
        raise RuntimeError("Action vocabulary must contain 23 unique actions.")
    if len(base_feature_columns()) != 60:
        raise RuntimeError("The 35-day X_base schema must contain 60 features.")
    if abs(sum(DATASET_CONTRACT.target_user_ratios) - 1.0) > 1e-12:
        raise RuntimeError("Target split ratios must sum to one.")
    if (
        len(EXPERIMENT_SEEDS) != 5
        or len(set(EXPERIMENT_SEEDS)) != 5
        or any(seed <= 0 for seed in EXPERIMENT_SEEDS)
    ):
        raise RuntimeError("The experiment must use five unique seeds.")


_validate_contract()
