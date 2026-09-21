import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import sparse
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_data import LocalHypergraph, batch_local_hypergraphs
from hsl import (
    classification_loss,
    hgsl_objective,
    sample_incident_nodes,
    train_pos_weight,
)
from paths import REPORTS_DIR
from train import HGSLTrainingConfig


class PhaseEightUnitTests(unittest.TestCase):
    def test_objective_is_bce_plus_weighted_contrastive_loss(self):
        logits = torch.tensor([0.2, -0.4, 0.7, -0.1])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        z0 = torch.eye(4)
        z_star = z0[torch.tensor([1, 0, 3, 2])]
        pos_weight = train_pos_weight(labels)
        total, parts = hgsl_objective(
            logits,
            labels,
            z0,
            z_star,
            pos_weight,
            contrastive_weight=0.25,
            temperature=0.2,
        )
        torch.testing.assert_close(
            parts["bce"], classification_loss(logits, labels, pos_weight)
        )
        torch.testing.assert_close(
            total, parts["bce"] + 0.25 * parts["contrastive"]
        )

    def test_local_batch_sampling_never_crosses_graphs(self):
        graph0 = self._local_graph(10, 11)
        graph1 = self._local_graph(20, 21)
        batch = batch_local_hypergraphs([graph0, graph1])
        sampled = sample_incident_nodes(
            batch.incidence,
            np.arange(batch.incidence.shape[1]),
            positive_count=1,
            negative_count=1,
            generator=np.random.default_rng(8),
            node_groups=batch.node_graph_ids,
            edge_groups=batch.edge_graph_ids,
        )
        for edge, positive, negative in zip(
            sampled.edge_ids,
            sampled.positive_nodes,
            sampled.negative_nodes,
            strict=True,
        ):
            expected_group = batch.edge_graph_ids[edge]
            self.assertTrue(np.all(batch.node_graph_ids[positive] == expected_group))
            self.assertTrue(np.all(batch.node_graph_ids[negative] == expected_group))

    def test_training_config_rejects_invalid_protocol(self):
        with self.assertRaises(ValueError):
            HGSLTrainingConfig(seed=42).validate()
        with self.assertRaises(ValueError):
            HGSLTrainingConfig(contrastive_weight=-0.1).validate()
        with self.assertRaises(ValueError):
            HGSLTrainingConfig(validation_limit=1).validate()

    @staticmethod
    def _local_graph(target: int, reference: int) -> LocalHypergraph:
        return LocalHypergraph(
            node_ids=np.array([target, reference], dtype=np.int64),
            features=np.zeros((2, 3), dtype=np.float32),
            target_mask=np.array([True, False]),
            incidence=sparse.csr_matrix(np.array([[1], [1]], dtype=np.uint8)),
            hyperedges=({"family": "course"},),
        )


@unittest.skipUnless(
    (REPORTS_DIR / "phase8_hgsl_full_seed_1.json").is_file(),
    "Phase 8 training report has not been generated",
)
class PhaseEightIntegrationTests(unittest.TestCase):
    def test_real_training_report_and_checkpoint(self):
        report = json.loads(
            (REPORTS_DIR / "phase8_hgsl_full_seed_1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["phase"], 8)
        self.assertEqual(report["config"]["feature_set"], "full")
        self.assertEqual(report["feature_dim"], 96)
        self.assertEqual(report["selection_metric"], "validation_auc")
        self.assertFalse(report["test_split_used"])
        self.assertEqual(report["local_negative_scope"], "same local graph only")
        self.assertGreater(report["epochs_completed"], 0)
        self.assertGreater(report["history"][0]["loss"]["total"], 0)
        self.assertGreater(report["history"][0]["scorer_gradient_norm"], 0)
        self.assertTrue(Path(report["checkpoint"]).is_file())
        self.assertEqual(len(report["graph_manifest_sha256"]), 64)
        self.assertEqual(len(report["feature_manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
