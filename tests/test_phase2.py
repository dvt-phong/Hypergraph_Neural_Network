from pathlib import Path
import sys
import unittest

import duckdb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import DATASET_CONTRACT, EXPERIMENT_SEEDS, EXPERIMENT_SPLITS
from split import UserGroup, assign_user_groups, integer_targets
from paths import PROCESSED_DATA_DIR


class PhaseTwoUnitTests(unittest.TestCase):
    def test_five_experiment_seeds_are_locked(self):
        self.assertEqual(EXPERIMENT_SEEDS, (1, 11, 111, 1111, 11111))
        self.assertEqual(DATASET_CONTRACT.experiment_seeds, EXPERIMENT_SEEDS)

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
        first, targets, actual = assign_user_groups(groups, seed=111)
        second, _, _ = assign_user_groups(groups, seed=111)
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
    (PROCESSED_DATA_DIR / "splits.parquet").is_file(),
    "Phase 2 artifacts have not been built",
)
class PhaseTwoIntegrationTests(unittest.TestCase):
    def test_split_artifact(self):
        path = (PROCESSED_DATA_DIR / "splits.parquet").as_posix()
        connection = duckdb.connect()
        try:
            columns = connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{path}')"
            ).fetchall()
            self.assertEqual(
                [column[0] for column in columns],
                [
                    "seed",
                    "node_id",
                    "enroll_id",
                    "user_id",
                    "source_partition",
                    "experiment_split",
                ],
            )
            invariants = connection.execute(
                f"""
                SELECT seed, count(*) AS rows,
                       count(DISTINCT node_id) AS nodes,
                       count(DISTINCT user_id) AS users
                FROM read_parquet('{path}')
                GROUP BY seed ORDER BY seed
                """
            ).fetchall()
            self.assertEqual([row[0] for row in invariants], list(EXPERIMENT_SEEDS))
            for _, rows, nodes, users in invariants:
                self.assertEqual(rows, DATASET_CONTRACT.enrollments)
                self.assertEqual(nodes, DATASET_CONTRACT.enrollments)
                self.assertEqual(users, DATASET_CONTRACT.users)

            overlap = connection.execute(
                f"""
                SELECT count(*) FROM (
                    SELECT seed, user_id
                    FROM read_parquet('{path}')
                    GROUP BY seed, user_id
                    HAVING count(DISTINCT experiment_split) > 1
                )
                """
            ).fetchone()[0]
            self.assertEqual(overlap, 0)

            changed = connection.execute(
                f"""
                WITH base AS (
                    SELECT user_id, experiment_split
                    FROM read_parquet('{path}') WHERE seed=1
                )
                SELECT s.seed,
                       count(*) FILTER (
                           WHERE s.experiment_split<>b.experiment_split
                       ) AS changed_users
                FROM read_parquet('{path}') s JOIN base b USING(user_id)
                WHERE s.seed<>1 GROUP BY s.seed
                """
            ).fetchall()
            self.assertEqual({int(row[0]) for row in changed}, set(EXPERIMENT_SEEDS[1:]))
            self.assertTrue(all(int(row[1]) > 0 for row in changed))
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
