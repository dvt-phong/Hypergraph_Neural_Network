import json
from pathlib import Path
import sys
import unittest

import duckdb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.schema import DATASET_CONTRACT, EXPERIMENT_SEEDS
from hypergraph.behavioral import K_CANDIDATES, K_MAX, build_seed_neighbors
from hypergraph.construction import build_hyperedges
from hypergraph.object import OBJECT_TYPES
from paths import PROCESSED_DATA_DIR


class PhaseFourUnitTests(unittest.TestCase):
    def test_locked_hyperedge_contract(self):
        self.assertEqual(OBJECT_TYPES, ("video", "assignment", "forum"))
        self.assertEqual(K_CANDIDATES, (5, 10, 20))

    def test_behavioral_search_uses_train_references_only(self):
        generator = np.random.default_rng(7)
        features = generator.normal(size=(40, 6)).astype(np.float32)
        train_ids = np.arange(30, dtype=np.int64)
        neighbors = build_seed_neighbors(features, train_ids, k_max=5)
        self.assertEqual(neighbors.shape, (40, 5))
        self.assertTrue(np.all(neighbors < 30))
        self.assertFalse(
            np.any(neighbors[train_ids] == train_ids[:, None])
        )


@unittest.skipUnless(
    (PROCESSED_DATA_DIR / "hyperedge_manifest.json").is_file(),
    "Phase 4 artifacts have not been built",
)
class PhaseFourIntegrationTests(unittest.TestCase):
    def test_structural_membership_contract(self):
        memberships = (
            PROCESSED_DATA_DIR / "structural_memberships.parquet"
        ).as_posix()
        connection = duckdb.connect()
        try:
            rows = connection.execute(
                f"""
                SELECT
                    count(*) FILTER (WHERE family='course') AS course_rows,
                    count(*) FILTER (WHERE family='object') AS object_rows,
                    count(*) FILTER (
                        WHERE family='object' AND object_id IS NULL
                    ) AS missing_objects,
                    count(DISTINCT object_type) FILTER (
                        WHERE family='object'
                    ) AS object_types
                FROM read_parquet('{memberships}')
                """
            ).fetchone()
            conflicts = connection.execute(
                f"""
                SELECT count(*) FROM (
                    SELECT course_id, object_id
                    FROM read_parquet('{memberships}')
                    WHERE family='object'
                    GROUP BY course_id, object_id
                    HAVING count(DISTINCT object_type) > 1
                )
                """
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(rows[0], DATASET_CONTRACT.enrollments)
        self.assertGreater(rows[1], 0)
        self.assertEqual(rows[2], 0)
        self.assertEqual(rows[3], 3)
        self.assertEqual(conflicts, 0)

    def test_behavioral_neighbors_are_train_only(self):
        with np.load(
            PROCESSED_DATA_DIR / "behavioral_neighbors.npz", allow_pickle=False
        ) as archive:
            neighbors = archive["neighbors"]
        self.assertEqual(
            neighbors.shape,
            (len(EXPERIMENT_SEEDS), DATASET_CONTRACT.enrollments, K_MAX),
        )
        self.assertEqual(neighbors.dtype, np.int32)

        splits = (PROCESSED_DATA_DIR / "splits.parquet").as_posix()
        connection = duckdb.connect()
        try:
            for seed_index, seed in enumerate(EXPERIMENT_SEEDS):
                rows = connection.execute(
                    f"""SELECT node_id FROM read_parquet('{splits}')
                        WHERE seed=? AND experiment_split='train'""",
                    [seed],
                ).fetchnumpy()
                train_ids = rows["node_id"].astype(np.int64, copy=False)
                train_mask = np.zeros(DATASET_CONTRACT.enrollments, dtype=bool)
                train_mask[train_ids] = True
                self.assertTrue(np.all(train_mask[neighbors[seed_index]]))
                self.assertFalse(
                    np.any(
                        neighbors[seed_index, train_ids]
                        == train_ids[:, None]
                    )
                )
        finally:
            connection.close()

    def test_audit_and_cache(self):
        audit = json.loads(
            (PROCESSED_DATA_DIR / "hyperedge_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            set(audit["structural"]["global"]["object_by_type"]),
            set(OBJECT_TYPES),
        )
        self.assertEqual(
            set(audit["behavioral_train_by_seed"]),
            {str(seed) for seed in EXPERIMENT_SEEDS},
        )
        manifest = build_hyperedges()
        self.assertTrue(manifest["cache_hit"])


if __name__ == "__main__":
    unittest.main()
