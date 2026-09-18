import json
from pathlib import Path
import sys
import unittest

import duckdb
import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.schema import DATASET_CONTRACT, EXPERIMENT_SEEDS, base_feature_columns
from features.engineering import ABLATION_FEATURES
from features.transform import build_features
from paths import PROCESSED_DATA_DIR


class PhaseThreeUnitTests(unittest.TestCase):
    def test_feature_contract(self):
        columns = base_feature_columns()
        self.assertEqual(len(columns), 60)
        self.assertEqual(len(ABLATION_FEATURES), 8)
        self.assertEqual(columns[-2:], ("session_count", "distinct_observed_objects"))


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
            manifest["array_shape"],
            [len(EXPERIMENT_SEEDS), DATASET_CONTRACT.enrollments, 60],
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

        matrix = np.load(PROCESSED_DATA_DIR / "X_base.npy", mmap_mode="r")
        self.assertEqual(matrix.shape, tuple(manifest["array_shape"]))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertTrue(np.isfinite(matrix[:, ::1000, :]).all())

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


if __name__ == "__main__":
    unittest.main()
