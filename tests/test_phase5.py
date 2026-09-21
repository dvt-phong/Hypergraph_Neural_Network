import json
from pathlib import Path
import sys
import unittest

import duckdb
import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import EXPERIMENT_SEEDS
from hypergraph.behavioral import DEFAULT_K, K_CANDIDATES
from hypergraph.construction import build_initial_hypergraph
from graph_data import load_local_hypergraph, load_train_hypergraph
from paths import PROCESSED_DATA_DIR


class PhaseFiveUnitTests(unittest.TestCase):
    def test_default_behavioral_k_is_a_candidate(self):
        self.assertEqual(DEFAULT_K, 10)
        self.assertIn(DEFAULT_K, K_CANDIDATES)

    def test_invalid_behavioral_k_is_rejected(self):
        with self.assertRaises(ValueError):
            build_initial_hypergraph(behavioral_k=7)


@unittest.skipUnless(
    (PROCESSED_DATA_DIR / "hypergraph_manifest.json").is_file(),
    "Phase 5 artifacts have not been built",
)
class PhaseFiveIntegrationTests(unittest.TestCase):
    def test_train_h0_is_binary_sparse_and_matches_metadata(self):
        metadata_path = (PROCESSED_DATA_DIR / "hyperedges.parquet").as_posix()
        connection = duckdb.connect()
        try:
            for seed in EXPERIMENT_SEEDS:
                matrix = sparse.load_npz(
                    PROCESSED_DATA_DIR / f"H0_train_seed_{seed}.npz"
                )
                self.assertTrue(sparse.isspmatrix_csr(matrix))
                self.assertEqual(matrix.dtype, np.uint8)
                self.assertTrue(np.all(matrix.data == 1))
                sizes = connection.execute(
                    f"""SELECT size FROM read_parquet('{metadata_path}')
                        WHERE seed=? ORDER BY hyperedge_id""",
                    [seed],
                ).fetchnumpy()["size"]
                self.assertEqual(matrix.shape[1], sizes.size)
                self.assertGreaterEqual(int(sizes.min()), 2)
                np.testing.assert_array_equal(
                    np.asarray(matrix.sum(axis=0)).reshape(-1), sizes
                )
        finally:
            connection.close()

    def test_train_node_index_matches_matrix_rows(self):
        index_path = (
            PROCESSED_DATA_DIR / "train_node_index.parquet"
        ).as_posix()
        connection = duckdb.connect()
        try:
            for seed in EXPERIMENT_SEEDS:
                summary = connection.execute(
                    f"""SELECT count(*), min(local_node_id), max(local_node_id),
                               count(DISTINCT node_id)
                        FROM read_parquet('{index_path}') WHERE seed=?""",
                    [seed],
                ).fetchone()
                matrix = sparse.load_npz(
                    PROCESSED_DATA_DIR / f"H0_train_seed_{seed}.npz"
                )
                self.assertEqual(summary[0], matrix.shape[0])
                self.assertEqual(summary[1], 0)
                self.assertEqual(summary[2], matrix.shape[0] - 1)
                self.assertEqual(summary[3], matrix.shape[0])
        finally:
            connection.close()

    def test_local_graphs_have_one_target_and_train_references(self):
        splits = (PROCESSED_DATA_DIR / "splits.parquet").as_posix()
        metadata = (PROCESSED_DATA_DIR / "hyperedges.parquet").as_posix()
        connection = duckdb.connect()
        try:
            for split in ("validation", "test"):
                path = (
                    PROCESSED_DATA_DIR / f"{split}_memberships.parquet"
                ).as_posix()
                invalid_targets = connection.execute(
                    f"""
                    SELECT count(*) FROM (
                        SELECT m.seed, m.target_node_id
                        FROM read_parquet('{path}') m
                        JOIN read_parquet('{splits}') s
                          ON s.seed=m.seed AND s.node_id=m.target_node_id
                        GROUP BY m.seed, m.target_node_id
                        HAVING min(s.experiment_split)!='{split}'
                            OR count(*) FILTER (WHERE family='course') != 1
                            OR count(*) FILTER (WHERE family='behavioral') != 1
                            OR min(size) < 2
                    )
                    """
                ).fetchone()[0]
                invalid_references = connection.execute(
                    f"""
                    SELECT count(*)
                    FROM read_parquet('{path}') m
                    LEFT JOIN read_parquet('{splits}') s
                      ON s.seed=m.seed
                     AND s.node_id=m.singleton_reference_node_id
                    LEFT JOIN read_parquet('{metadata}') h
                      ON h.seed=m.seed AND h.hyperedge_id=m.train_hyperedge_id
                    WHERE (m.singleton_reference_node_id IS NOT NULL
                           AND s.experiment_split IS DISTINCT FROM 'train')
                       OR (m.train_hyperedge_id IS NOT NULL
                           AND (h.family IS DISTINCT FROM m.family
                                OR h.family='behavioral'))
                    """
                ).fetchone()[0]
                self.assertEqual(invalid_targets, 0)
                self.assertEqual(invalid_references, 0)
        finally:
            connection.close()

    def test_second_build_uses_cache(self):
        manifest = build_initial_hypergraph(behavioral_k=DEFAULT_K)
        self.assertTrue(manifest["cache_hit"])
        stored = json.loads(
            (PROCESSED_DATA_DIR / "hypergraph_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(stored["behavioral_k"], DEFAULT_K)

    def test_graph_loaders_preserve_feature_and_target_order(self):
        train = load_train_hypergraph(EXPERIMENT_SEEDS[0])
        self.assertEqual(train.incidence.shape[0], train.node_ids.size)
        self.assertEqual(train.features.shape, (train.node_ids.size, 60))

        splits = (PROCESSED_DATA_DIR / "splits.parquet").as_posix()
        connection = duckdb.connect()
        try:
            targets = {
                split: connection.execute(
                    f"""SELECT min(node_id) FROM read_parquet('{splits}')
                        WHERE seed=? AND experiment_split=?""",
                    [EXPERIMENT_SEEDS[0], split],
                ).fetchone()[0]
                for split in ("validation", "test")
            }
        finally:
            connection.close()
        for split, target in targets.items():
            local = load_local_hypergraph(EXPERIMENT_SEEDS[0], split, target)
            self.assertEqual(local.node_ids[0], target)
            self.assertEqual(int(local.target_mask.sum()), 1)
            self.assertTrue(local.target_mask[0])
            self.assertEqual(local.features.shape, (local.node_ids.size, 60))
            self.assertTrue(np.all(local.incidence.getrow(0).toarray() == 1))
            self.assertGreaterEqual(
                int(np.asarray(local.incidence.sum(axis=0)).min()), 2
            )

    def test_graph_loaders_support_all_feature_sets(self):
        expected = {
            "behavior": 60,
            "behavior_user": 75,
            "behavior_course": 81,
            "full": 96,
        }
        seed = EXPERIMENT_SEEDS[0]
        for feature_set, dimension in expected.items():
            train = load_train_hypergraph(seed, feature_set=feature_set)
            self.assertEqual(train.features.shape, (train.node_ids.size, dimension))
            self.assertTrue(np.isfinite(train.features[::1000]).all())


if __name__ == "__main__":
    unittest.main()
