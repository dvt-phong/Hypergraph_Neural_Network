from contextlib import redirect_stdout
from importlib.util import find_spec
from io import StringIO
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main import main
from config import (
    ACTIONS,
    ACTION_GROUPS,
    DATASET_CONTRACT,
    EARLY_OBSERVATION_DAYS,
    OBSERVATION_DAYS,
    base_feature_columns,
    context_feature_columns,
    course_feature_columns,
    feature_columns,
    user_feature_columns,
)
from paths import PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, require_project_path


class PhaseZeroTests(unittest.TestCase):
    def test_code_is_directly_under_src(self):
        self.assertFalse((ROOT / "src" / "mooc_hgsl").exists())
        for package in ("data", "features", "hypergraph"):
            self.assertIsNotNone(find_spec(package), package)
        for module in (
            "main",
            "config",
            "artifacts",
            "graph_data",
            "model",
            "hsl",
            "train",
            "metrics",
        ):
            self.assertIsNotNone(find_spec(module), module)

    def test_dataset_scope_is_locked(self):
        self.assertEqual(DATASET_CONTRACT.dataset, "XuetangX-247")
        self.assertEqual(DATASET_CONTRACT.enrollments, 225_642)
        self.assertEqual(DATASET_CONTRACT.users, 77_083)
        self.assertEqual(DATASET_CONTRACT.courses, 247)
        self.assertEqual(
            DATASET_CONTRACT.main_hyperedge_families,
            ("course", "object", "behavioral"),
        )

    def test_action_vocabulary_is_unique_and_complete(self):
        grouped = tuple(action for group in ACTION_GROUPS.values() for action in group)
        self.assertEqual(grouped, ACTIONS)
        self.assertEqual(len(ACTIONS), 23)
        self.assertEqual(len(set(ACTIONS)), 23)

    def test_main_feature_schema_has_60_columns(self):
        self.assertEqual(OBSERVATION_DAYS, 35)
        columns = base_feature_columns()
        self.assertEqual(len(columns), 60)
        self.assertEqual(columns[:2], ("day_00", "day_01"))
        self.assertEqual(columns[-2:], ("session_count", "distinct_observed_objects"))

    def test_context_feature_schemas_have_fixed_dimensions(self):
        self.assertEqual(len(user_feature_columns()), 15)
        self.assertEqual(len(course_feature_columns()), 21)
        self.assertEqual(len(context_feature_columns()), 36)
        self.assertEqual(len(feature_columns("behavior")), 60)
        self.assertEqual(len(feature_columns("behavior_user")), 75)
        self.assertEqual(len(feature_columns("behavior_course")), 81)
        self.assertEqual(len(feature_columns("full")), 96)
        self.assertEqual(len(set(feature_columns("full"))), 96)
        with self.assertRaises(ValueError):
            feature_columns("unknown")

    def test_early_windows_use_the_same_action_schema(self):
        for days in EARLY_OBSERVATION_DAYS:
            self.assertEqual(len(base_feature_columns(days)), days + 23 + 2)
        with self.assertRaises(ValueError):
            base_feature_columns(10)

    def test_paths_are_stable_and_inside_project(self):
        self.assertEqual(PROJECT_ROOT, ROOT)
        self.assertEqual(RAW_DATA_DIR, ROOT / "data" / "raw" / "xuetangx")
        self.assertEqual(PROCESSED_DATA_DIR, ROOT / "data" / "processed" / "xuetangx_247")
        self.assertEqual(require_project_path(PROCESSED_DATA_DIR), PROCESSED_DATA_DIR.resolve())
        with self.assertRaises(ValueError):
            require_project_path(ROOT.parent)

    def test_contract_cli_is_machine_readable(self):
        output = StringIO()
        with redirect_stdout(output):
            exit_code = main(["contract"])
        contract = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(contract["schema_version"], "xuetangx-247-v1")
        self.assertEqual(contract["courses"], 247)

    def test_project_has_no_external_config_directory(self):
        self.assertFalse((ROOT / "configs").exists())


if __name__ == "__main__":
    unittest.main()
