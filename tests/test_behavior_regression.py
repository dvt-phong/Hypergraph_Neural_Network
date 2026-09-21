"""Small deterministic checks that protect model behavior during refactoring."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import sparse
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hsl import (
    HGSLModel,
    RefinementConfig,
    classification_loss,
    hgsl_objective,
    train_pos_weight,
)
from model import HGNNBaseline, HypergraphOperator


class ModelBehaviorRegressionTests(unittest.TestCase):
    def test_baseline_forward_values_are_unchanged(self):
        incidence = sparse.csr_matrix(
            np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)
        )
        operator = HypergraphOperator.from_scipy(
            incidence, device=torch.device("cpu")
        )
        torch.manual_seed(1)
        model = HGNNBaseline(input_dim=4, hidden_dim=8, dropout=0.0)
        features = torch.randn(3, 4)
        labels = torch.tensor([0.0, 1.0, 1.0])

        logits, embeddings = model(features, operator)
        loss = classification_loss(
            logits, labels, train_pos_weight(labels)
        )

        torch.testing.assert_close(
            logits,
            torch.tensor([-0.23696540, -0.25590780, -0.25881818]),
        )
        self.assertAlmostEqual(float(embeddings.detach().sum()), 0.85511196, places=6)
        self.assertAlmostEqual(float(loss.detach()), 0.47058427, places=6)

    def test_hgsl_refinement_values_are_unchanged(self):
        incidence = sparse.csr_matrix(
            np.array(
                [
                    [1, 0, 1, 0, 0, 0],
                    [1, 0, 1, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 1, 0, 0, 1, 1],
                    [0, 1, 0, 0, 0, 1],
                ],
                dtype=np.uint8,
            )
        )
        families = np.array(
            ["course", "course", "object", "object", "behavioral", "behavioral"]
        )
        sizes = np.asarray(incidence.sum(axis=0)).reshape(-1)
        labels = torch.tensor(
            [0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32
        )

        torch.manual_seed(7)
        features = torch.randn(8, 4)
        operator = HypergraphOperator.from_scipy(
            incidence, device=torch.device("cpu")
        )
        model = HGSLModel(
            input_dim=4,
            hidden_dim=8,
            dropout=0.0,
            refinement=RefinementConfig(
                sampled_hyperedges=6,
                positive_nodes=2,
                negative_nodes=2,
                top_r=2,
            ),
        )

        output = model(
            features,
            operator,
            incidence,
            families,
            sizes,
            np.random.default_rng(11),
        )
        total_loss, loss_parts = hgsl_objective(
            output.logits,
            labels,
            output.z0,
            output.z_star,
            train_pos_weight(labels),
            contrastive_weight=0.1,
        )

        expected_indices = torch.tensor(
            [
                [0, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 7, 7],
                [0, 1, 2, 5, 0, 0, 3, 1, 0, 1, 2, 4, 5, 3, 4],
            ]
        )
        self.assertTrue(
            torch.equal(output.refinement.incidence.indices(), expected_indices)
        )
        self.assertAlmostEqual(float(output.z0.detach().sum()), 0.65764368, places=6)
        self.assertAlmostEqual(float(output.z_star.detach().sum()), 0.43438259, places=6)
        self.assertAlmostEqual(float(total_loss.detach()), 0.92247641, places=6)
        self.assertAlmostEqual(
            float(loss_parts["contrastive"].detach()), 2.22579908, places=6
        )


if __name__ == "__main__":
    unittest.main()
