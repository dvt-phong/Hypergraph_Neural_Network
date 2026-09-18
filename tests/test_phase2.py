import json
from pathlib import Path
import sys
import unittest

import duckdb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.schema import DATASET_CONTRACT, EXPERIMENT_SPLITS
from data.split import UserGroup, assign_user_groups, build_splits, integer_targets
from paths import PROCESSED_DATA_DIR


class PhaseTwoUnitTests(unittest.TestCase):
    def test_integer_targets_preserve_total(self):
        targets = integer_targets(77_083, (0.64, 0.16, 0.20))
        self.assertEqual(
            targets,
            {"train": 49_333, "validation": 12_333, "test": 15_417},
        )
        self.assertEqual(sum(targets.values()), 77_083)

    def test_assignment_is_deterministic_and_user_disjoint(self):
        groups = [
            UserGroup(user_id=index, enrollments=index % 5 + 1, dropouts=index % 2)
            for index in range(1, 101)
        ]
        first, targets, actual = assign_user_groups(groups, seed=42)
        second, _, _ = assign_user_groups(groups, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(groups))
        self.assertTrue(set(first.values()).issubset(EXPERIMENT_SPLITS))
        for split in EXPERIMENT_SPLITS:
            self.assertEqual(actual[split]["users"], targets[split]["users"])

    def test_invalid_user_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            assign_user_groups([])
        duplicate = [UserGroup(1, 1, 0), UserGroup(1, 2, 1)]
        with self.assertRaises(ValueError):
            assign_user_groups(duplicate)


@unittest.skipUnless(
    (PROCESSED_DATA_DIR / "split_manifest.json").is_file(),
    "Phase 2 artifacts have not been built",
)
class PhaseTwoIntegrationTests(unittest.TestCase):
    def test_split_artifact_and_manifest(self):
        manifest = json.loads(
            (PROCESSED_DATA_DIR / "split_manifest.json").read_text(encoding="utf-8")
        )
        invariants = manifest["invariants"]
        self.assertEqual(invariants["rows"], DATASET_CONTRACT.enrollments)
        self.assertEqual(invariants["distinct_users"], DATASET_CONTRACT.users)
        self.assertEqual(invariants["user_overlap"], 0)
        self.assertEqual(sum(row["users"] for row in manifest["summary"].values()), 77_083)
        self.assertEqual(
            sum(row["enrollments"] for row in manifest["summary"].values()),
            225_642,
        )
        for row in manifest["summary"].values():
            enrollment_error = abs(row["enrollment_deviation"]) / row[
                "target_enrollments"
            ]
            dropout_error = abs(row["dropout_deviation"]) / row["target_dropouts"]
            self.assertLess(enrollment_error, 0.005)
            self.assertLess(dropout_error, 0.005)

        path = (PROCESSED_DATA_DIR / "splits.parquet").as_posix()
        connection = duckdb.connect()
        try:
            columns = connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{path}')"
            ).fetchall()
            self.assertEqual(
                [column[0] for column in columns],
                [
                    "node_id",
                    "enroll_id",
                    "user_id",
                    "source_partition",
                    "experiment_split",
                ],
            )
        finally:
            connection.close()

    def test_second_build_uses_cache(self):
        manifest = build_splits()
        self.assertTrue(manifest["cache_hit"])


if __name__ == "__main__":
    unittest.main()
