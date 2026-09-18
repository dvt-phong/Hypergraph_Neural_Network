import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import sparse
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from losses.objective import classification_loss, train_pos_weight
from models.hgnn import HypergraphOperator
from models.model import HGNNBaseline
from paths import REPORTS_DIR
from training.evaluator import binary_metrics
from training.trainer import BaselineConfig


class PhaseSixUnitTests(unittest.TestCase):
    def setUp(self):
        self.incidence = sparse.csr_matrix(
            np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)
        )
        self.operator = HypergraphOperator.from_scipy(
            self.incidence, device=torch.device("cpu")
        )

    def test_sparse_propagation_matches_dense_definition(self):
        features = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )
        actual = self.operator.propagate(features).numpy()
        h = self.incidence.toarray().astype(np.float32)
        edge_inverse_degree = np.diag(1 / h.sum(axis=0))
        node_inverse_sqrt_degree = np.diag(1 / np.sqrt(h.sum(axis=1)))
        propagation = (
            node_inverse_sqrt_degree
            @ h
            @ edge_inverse_degree
            @ h.T
            @ node_inverse_sqrt_degree
        )
        np.testing.assert_allclose(actual, propagation @ features.numpy(), rtol=1e-6)

    def test_forward_loss_and_backward_are_finite(self):
        torch.manual_seed(1)
        model = HGNNBaseline(input_dim=4, hidden_dim=8, dropout=0.0)
        features = torch.randn(3, 4)
        labels = torch.tensor([0.0, 1.0, 1.0])
        logits, embeddings = model(features, self.operator)
        self.assertEqual(logits.shape, (3,))
        self.assertEqual(embeddings.shape, (3, 8))
        loss = classification_loss(logits, labels, train_pos_weight(labels))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_pos_weight_and_metrics(self):
        labels = torch.tensor([0.0, 0.0, 1.0])
        self.assertAlmostEqual(float(train_pos_weight(labels)), 2.0)
        metrics = binary_metrics(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
        )
        for value in metrics.values():
            self.assertAlmostEqual(value, 1.0)
        reference = binary_metrics(
            np.array([0, 1, 1, 0]), np.array([0.1, 0.4, 0.35, 0.8])
        )
        self.assertAlmostEqual(reference["auc"], 0.5)
        self.assertAlmostEqual(reference["auprc"], 7 / 12)
        tied = binary_metrics(np.array([1, 0]), np.array([0.5, 0.5]))
        self.assertAlmostEqual(tied["auc"], 0.5)
        self.assertAlmostEqual(tied["auprc"], 0.5)

    def test_config_rejects_non_experiment_seed(self):
        with self.assertRaises(ValueError):
            BaselineConfig(seed=42).validate()


@unittest.skipUnless(
    (REPORTS_DIR / "phase6_hgnn_seed_1.json").is_file(),
    "Phase 6 smoke report has not been generated",
)
class PhaseSixIntegrationTests(unittest.TestCase):
    def test_real_smoke_report(self):
        report = json.loads(
            (REPORTS_DIR / "phase6_hgnn_seed_1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["phase"], 6)
        self.assertEqual(report["config"]["seed"], 1)
        self.assertEqual(report["nodes"], 144543)
        self.assertEqual(report["feature_dim"], 60)
        self.assertGreater(report["hyperedges"], 100000)
        self.assertGreater(report["incidences"], 3000000)
        self.assertGreater(report["pos_weight"], 0)
        self.assertTrue(np.isfinite(report["loss_by_epoch"]).all())
        self.assertTrue(np.isfinite(report["gradient_norm_by_epoch"]).all())
        self.assertEqual(
            set(report["train_metrics"]),
            {"auc", "auprc", "f1", "precision", "recall"},
        )
        self.assertGreaterEqual(report["best_epoch"], 1)
        self.assertGreaterEqual(report["validation_targets"], 2)
        self.assertTrue(Path(report["checkpoint"]).is_file())
        self.assertEqual(
            set(report["best_validation_metrics"]) - {"epoch"},
            {"auc", "auprc", "f1", "precision", "recall"},
        )


if __name__ == "__main__":
    unittest.main()
