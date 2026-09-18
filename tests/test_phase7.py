import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import sparse
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from losses.contrastive import contrastive_alignment_loss
from losses.objective import classification_loss, hgsl_objective, train_pos_weight
from models.hgnn import HypergraphOperator, sparse_values_mm
from models.hyperedge_sampling import sample_balanced_hyperedges, sampled_strata_audit
from models.incident_node_sampling import sample_incident_nodes
from models.model import HGSLModel
from models.refinement import RefinementConfig
from paths import REPORTS_DIR


class PhaseSevenUnitTests(unittest.TestCase):
    def setUp(self):
        self.incidence = sparse.csr_matrix(
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
        self.families = np.array(
            ["course", "course", "object", "object", "behavioral", "behavioral"]
        )
        self.sizes = np.asarray(self.incidence.sum(axis=0)).reshape(-1)

    def test_balanced_hyperedge_sampling(self):
        families = np.repeat(["course", "object", "behavioral"], 6)
        sizes = np.tile([2, 3, 20, 30, 200, 300], 3)
        selected = sample_balanced_hyperedges(
            families, sizes, 9, np.random.default_rng(3)
        )
        audit = sampled_strata_audit(selected, families, sizes)
        self.assertEqual(len(audit), 9)
        self.assertEqual(set(audit.values()), {1})

    def test_incident_negative_sampling_has_no_overlap(self):
        sampled = sample_incident_nodes(
            self.incidence,
            np.arange(self.incidence.shape[1]),
            positive_count=2,
            negative_count=3,
            generator=np.random.default_rng(5),
        )
        csc = self.incidence.tocsc()
        for edge_id, positive, negative in zip(
            sampled.edge_ids,
            sampled.positive_nodes,
            sampled.negative_nodes,
            strict=True,
        ):
            start, stop = csc.indptr[edge_id : edge_id + 2]
            incident = csc.indices[start:stop]
            self.assertEqual(np.intersect1d(positive, negative).size, 0)
            self.assertEqual(np.intersect1d(incident, negative).size, 0)

    def test_hgsl_forward_is_sparse_and_differentiable(self):
        torch.manual_seed(7)
        features = torch.randn(8, 4)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
        initial_operator = HypergraphOperator.from_scipy(
            self.incidence, device=torch.device("cpu")
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
            initial_operator,
            self.incidence,
            self.families,
            self.sizes,
            np.random.default_rng(11),
        )
        self.assertTrue(output.refinement.incidence.is_sparse)
        self.assertEqual(output.logits.shape, (8,))
        self.assertEqual(output.z0.shape, output.z_star.shape)
        indices = output.refinement.incidence.indices()
        self.assertEqual(torch.unique(indices[1]).numel(), self.incidence.shape[1])
        self.assertEqual(torch.unique(indices[0]).numel(), self.incidence.shape[0])
        total, _ = hgsl_objective(
            output.logits,
            labels,
            output.z0,
            output.z_star,
            train_pos_weight(labels),
            contrastive_weight=0.1,
        )
        total.backward()
        scorer_gradients = [
            parameter.grad for parameter in model.refiner.scorer.parameters()
        ]
        self.assertTrue(
            all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in scorer_gradients
            )
        )
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in scorer_gradients), 0
        )

    def test_sparse_value_backward_matches_dense(self):
        indices = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
        sparse_values = torch.tensor(
            [0.4, 0.6, 0.7, 0.3], requires_grad=True
        )
        sparse_features = torch.randn(2, 3, requires_grad=True)
        matrix = torch.sparse_coo_tensor(
            indices, sparse_values, size=(3, 2), check_invariants=True
        ).coalesce()
        sparse_output = sparse_values_mm(matrix, sparse_features)
        sparse_loss = sparse_output.square().sum()
        sparse_loss.backward()

        dense_values = sparse_values.detach().clone().requires_grad_(True)
        dense_features = sparse_features.detach().clone().requires_grad_(True)
        dense_matrix = torch.zeros(3, 2).index_put(
            (indices[0], indices[1]), dense_values, accumulate=True
        )
        dense_output = dense_matrix @ dense_features
        dense_output.square().sum().backward()
        torch.testing.assert_close(sparse_output, dense_output)
        torch.testing.assert_close(sparse_values.grad, dense_values.grad)
        torch.testing.assert_close(sparse_features.grad, dense_features.grad)

    def test_contrastive_alignment_and_lambda_zero(self):
        aligned = torch.eye(4)
        shuffled = aligned[torch.tensor([1, 0, 3, 2])]
        aligned_loss = contrastive_alignment_loss(aligned, aligned)
        shuffled_loss = contrastive_alignment_loss(aligned, shuffled)
        self.assertLess(float(aligned_loss), float(shuffled_loss))

        logits = torch.tensor([0.1, -0.2, 0.3, -0.4])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        pos_weight = train_pos_weight(labels)
        expected = classification_loss(logits, labels, pos_weight)
        actual, parts = hgsl_objective(
            logits,
            labels,
            aligned,
            shuffled,
            pos_weight,
            contrastive_weight=0.0,
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(float(parts["contrastive"]), 0.0)


@unittest.skipUnless(
    (REPORTS_DIR / "phase7_hgsl_seed_1.json").is_file(),
    "Phase 7 smoke report has not been generated",
)
class PhaseSevenIntegrationTests(unittest.TestCase):
    def test_real_hgsl_smoke_report(self):
        report = json.loads(
            (REPORTS_DIR / "phase7_hgsl_seed_1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["phase"], 7)
        self.assertTrue(report["gradient_finite"])
        self.assertGreater(report["gradient_norm"], 0)
        self.assertGreater(report["refinement"]["sampled_hyperedges"], 0)
        self.assertGreater(report["refinement"]["refined_incidences"], 0)
        self.assertGreater(report["loss"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
