import json
from pathlib import Path
import sys
import unittest

import duckdb
import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import (
    COURSE_CATEGORY_VALUES,
    DATASET_CONTRACT,
    EDUCATION_VALUES,
    EXPERIMENT_SEEDS,
    FEATURE_SETS,
    GENDER_VALUES,
    base_feature_columns,
    context_feature_columns,
    course_feature_columns,
    feature_columns,
    user_feature_columns,
)
from features.context import RAW_CONTEXT_COLUMNS
from features.engineering import ABLATION_FEATURES
from features.transform import build_features
from paths import PROCESSED_DATA_DIR


class PhaseThreeUnitTests(unittest.TestCase):
    def test_feature_contract(self):
        columns = base_feature_columns()
        self.assertEqual(len(columns), 60)
        self.assertEqual(len(ABLATION_FEATURES), 8)
        self.assertEqual(columns[-2:], ("session_count", "distinct_observed_objects"))
        self.assertEqual(len(user_feature_columns()), 15)
        self.assertEqual(len(course_feature_columns()), 21)
        self.assertEqual(len(context_feature_columns()), 36)
        self.assertEqual(
            {name: len(feature_columns(name)) for name in FEATURE_SETS},
            {
                "behavior": 60,
                "behavior_user": 75,
                "behavior_course": 81,
                "full": 96,
            },
        )


@unittest.skipUnless(
    (PROCESSED_DATA_DIR / "feature_manifest.json").is_file(),
    "Phase 3 artifacts have not been built",
)
class PhaseThreeIntegrationTests(unittest.TestCase):
    def test_raw_features_and_x_base(self):
        manifest = json.loads(
            (PROCESSED_DATA_DIR / "feature_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["feature_names"], list(base_feature_columns()))
        self.assertEqual(
            manifest["context_feature_names"], list(context_feature_columns())
        )
        self.assertEqual(
            manifest["array_shape"],
            [len(EXPERIMENT_SEEDS), DATASET_CONTRACT.enrollments, 60],
        )
        self.assertEqual(
            manifest["context_array_shape"],
            [len(EXPERIMENT_SEEDS), DATASET_CONTRACT.enrollments, 36],
        )
        audit = manifest["raw_feature_audit"]
        self.assertEqual(audit["rows"], DATASET_CONTRACT.enrollments)
        self.assertEqual(audit["events"], DATASET_CONTRACT.retained_events_35d)
        self.assertEqual(audit["bad_daily_sum"], 0)
        self.assertEqual(audit["bad_action_sum"], 0)

        raw_path = (PROCESSED_DATA_DIR / "features_raw.parquet").as_posix()
        connection = duckdb.connect()
        try:
            columns = connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{raw_path}')"
            ).fetchall()
        finally:
            connection.close()
        self.assertEqual(
            [column[0] for column in columns],
            ["node_id", *base_feature_columns(), *ABLATION_FEATURES],
        )

        context_path = (PROCESSED_DATA_DIR / "context_raw.parquet").as_posix()
        connection = duckdb.connect()
        try:
            context_columns = connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{context_path}')"
            ).fetchall()
        finally:
            connection.close()
        self.assertEqual(
            [column[0] for column in context_columns],
            ["node_id", *RAW_CONTEXT_COLUMNS],
        )

        matrix = np.load(PROCESSED_DATA_DIR / "X_base.npy", mmap_mode="r")
        self.assertEqual(matrix.shape, tuple(manifest["array_shape"]))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertTrue(np.isfinite(matrix[:, ::1000, :]).all())
        context = np.load(PROCESSED_DATA_DIR / "X_context.npy", mmap_mode="r")
        self.assertEqual(context.shape, tuple(manifest["context_array_shape"]))
        self.assertEqual(context.dtype, np.float32)
        self.assertTrue(np.isfinite(context[:, ::1000, :]).all())

        sample = np.asarray(context[0, ::1000])
        gender_stop = len(GENDER_VALUES) + 2
        education_start = gender_stop
        education_stop = education_start + len(EDUCATION_VALUES) + 2
        category_start = len(user_feature_columns())
        category_stop = category_start + len(COURSE_CATEGORY_VALUES) + 2
        np.testing.assert_array_equal(sample[:, :gender_stop].sum(axis=1), 1)
        np.testing.assert_array_equal(
            sample[:, education_start:education_stop].sum(axis=1), 1
        )
        np.testing.assert_array_equal(
            sample[:, category_start:category_stop].sum(axis=1), 1
        )

    def test_transform_is_fit_on_each_train_split(self):
        transform = joblib.load(PROCESSED_DATA_DIR / "feature_transform.joblib")
        matrix = np.load(PROCESSED_DATA_DIR / "X_base.npy", mmap_mode="r")
        splits = (PROCESSED_DATA_DIR / "splits.parquet").as_posix()
        connection = duckdb.connect()
        try:
            for seed_index, seed in enumerate(EXPERIMENT_SEEDS):
                rows = connection.execute(
                    f"""
                    SELECT node_id FROM read_parquet('{splits}')
                    WHERE seed=? AND experiment_split='train' ORDER BY node_id
                    """,
                    [seed],
                ).fetchnumpy()
                node_ids = rows["node_id"].astype(np.int64, copy=False)
                train = np.asarray(matrix[seed_index, node_ids])
                zero_names = transform["by_seed"][str(seed)][
                    "zero_variance_features"
                ]
                zero_indices = [
                    base_feature_columns().index(name) for name in zero_names
                ]
                variable = np.ones(60, dtype=bool)
                variable[zero_indices] = False
                means = np.mean(train, axis=0, dtype=np.float64)
                standard_deviations = np.std(train, axis=0, dtype=np.float64)
                self.assertLess(float(np.max(np.abs(means[variable]))), 2e-5)
                self.assertLess(
                    float(np.max(np.abs(standard_deviations[variable] - 1.0))),
                    2e-4,
                )
                if zero_indices:
                    self.assertTrue(np.all(train[:, zero_indices] == 0))
                del train
        finally:
            connection.close()

    def test_second_build_uses_cache(self):
        manifest = build_features()
        self.assertTrue(manifest["cache_hit"])

    def test_context_numeric_transform_is_fit_on_train_only(self):
        transform = joblib.load(
            PROCESSED_DATA_DIR / "context_transform.joblib"
        )
        matrix = np.load(PROCESSED_DATA_DIR / "X_context.npy", mmap_mode="r")
        age_index = user_feature_columns().index("age_at_course_start")
        duration_index = len(user_feature_columns()) + course_feature_columns().index(
            "course_duration_days"
        )
        splits = (PROCESSED_DATA_DIR / "splits.parquet").as_posix()
        connection = duckdb.connect()
        try:
            for seed_index, seed in enumerate(EXPERIMENT_SEEDS):
                node_ids = connection.execute(
                    f"""SELECT node_id FROM read_parquet('{splits}')
                        WHERE seed=? AND experiment_split='train'
                        ORDER BY node_id""",
                    [seed],
                ).fetchnumpy()["node_id"].astype(np.int64, copy=False)
                train = np.asarray(
                    matrix[seed_index, node_ids][:, [age_index, duration_index]]
                )
                self.assertLess(
                    float(np.max(np.abs(train.mean(axis=0, dtype=np.float64)))),
                    2e-5,
                )
                self.assertLess(
                    float(
                        np.max(
                            np.abs(train.std(axis=0, dtype=np.float64) - 1.0)
                        )
                    ),
                    2e-4,
                )
                self.assertEqual(
                    transform["by_seed"][str(seed)]["train_nodes"],
                    node_ids.size,
                )
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
