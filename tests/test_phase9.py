import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import sparse
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artifacts import source_signature
from features.transform import FEATURE_MANIFEST
from graph_data import LocalHypergraph, batch_local_hypergraphs
from hsl import HGSLModel, RefinementConfig
from hypergraph.construction import GRAPH_MANIFEST_ARTIFACT
from model import HypergraphOperator
from paths import REPORTS_DIR
from train import (
    HGSLTrainingConfig,
    configuration_id,
    experiment_id,
    evaluate_baseline_checkpoint,
    evaluate_hgsl_checkpoint,
    validate_checkpoint_artifacts,
)


class PhaseNineUnitTests(unittest.TestCase):
    def test_final_training_uses_full_validation_by_default(self):
        self.assertEqual(HGSLTrainingConfig().validation_limit, 0)

    def test_invalid_test_sampling_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_hgsl_checkpoint(
                seed=1,
                feature_set="full",
                test_limit=1,
            )
        with self.assertRaises(ValueError):
            evaluate_hgsl_checkpoint(
                seed=1,
                feature_set="full",
                batch_size=0,
            )
        with self.assertRaises(ValueError):
            evaluate_baseline_checkpoint(
                seed=1,
                feature_set="full",
                test_limit=1,
            )

    def test_configuration_id_prevents_hyperparameter_checkpoint_collision(self):
        first = HGSLTrainingConfig(top_r=4)
        second = HGSLTrainingConfig(top_r=8)
        same_model_on_cpu = HGSLTrainingConfig(top_r=4, device="cpu")
        self.assertNotEqual(configuration_id(first), configuration_id(second))
        self.assertEqual(configuration_id(first), configuration_id(same_model_on_cpu))

    def test_checkpoint_artifact_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            graph_manifest = output_dir / GRAPH_MANIFEST_ARTIFACT
            feature_manifest = output_dir / FEATURE_MANIFEST
            graph_manifest.write_text("graph-v1", encoding="utf-8")
            feature_manifest.write_text("feature-v1", encoding="utf-8")
            checkpoint = {
                "graph_manifest": source_signature(
                    graph_manifest, include_hash=True
                ),
                "feature_manifest": source_signature(
                    feature_manifest, include_hash=True
                ),
            }
            first_experiment = experiment_id(
                HGSLTrainingConfig(), graph_manifest, feature_manifest
            )
            validate_checkpoint_artifacts(checkpoint, output_dir)
            graph_manifest.write_text("graph-v2", encoding="utf-8")
            second_experiment = experiment_id(
                HGSLTrainingConfig(), graph_manifest, feature_manifest
            )
            self.assertNotEqual(first_experiment, second_experiment)
            with self.assertRaises(RuntimeError):
                validate_checkpoint_artifacts(checkpoint, output_dir)

    def test_deterministic_refinement_is_independent_of_batch_size(self):
        graph_one = LocalHypergraph(
            node_ids=np.array([10, 1, 2], dtype=np.int64),
            features=np.array(
                [[1.0, 0.0], [0.5, 1.0], [1.0, 0.5]], dtype=np.float32
            ),
            target_mask=np.array([True, False, False]),
            incidence=sparse.csr_matrix(
                np.array([[1, 1], [1, 0], [0, 1]], dtype=np.uint8)
            ),
            hyperedges=(
                {"family": "course"},
                {"family": "object"},
            ),
        )
        graph_two = LocalHypergraph(
            node_ids=np.array([20, 3, 4], dtype=np.int64),
            features=np.array(
                [[0.0, 1.0], [1.0, 1.0], [0.2, 0.8]], dtype=np.float32
            ),
            target_mask=np.array([True, False, False]),
            incidence=sparse.csr_matrix(
                np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)
            ),
            hyperedges=(
                {"family": "course"},
                {"family": "behavioral"},
            ),
        )
        torch.manual_seed(7)
        model = HGSLModel(
            input_dim=2,
            hidden_dim=4,
            dropout=0.0,
            refinement=RefinementConfig(
                sampled_hyperedges=1,
                positive_nodes=2,
                negative_nodes=1,
                top_r=2,
            ),
        )
        model.eval()

        def target_logits(graphs):
            batch = batch_local_hypergraphs(graphs)
            features = torch.as_tensor(batch.features, dtype=torch.float32)
            operator = HypergraphOperator.from_scipy(
                batch.incidence, device=torch.device("cpu")
            )
            with torch.no_grad():
                output = model(
                    features,
                    operator,
                    batch.incidence,
                    batch.families,
                    batch.sizes,
                    np.random.default_rng(1),
                    batch.node_graph_ids,
                    batch.edge_graph_ids,
                    deterministic=True,
                )
            return output.logits[torch.as_tensor(batch.target_indices)]

        separate = torch.cat(
            [target_logits([graph_one]), target_logits([graph_two])]
        )
        together = target_logits([graph_one, graph_two])
        torch.testing.assert_close(separate, together, rtol=1e-6, atol=1e-7)


@unittest.skipUnless(
    (REPORTS_DIR / "phase9_test_full_seed_1.json").is_file(),
    "Phase 9 test report has not been generated",
)
class PhaseNineIntegrationTests(unittest.TestCase):
    def test_test_report_keeps_model_selection_on_validation(self):
        report = json.loads(
            (REPORTS_DIR / "phase9_test_full_seed_1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["phase"], 9)
        self.assertEqual(report["checkpoint_selection_split"], "validation")
        self.assertFalse(report["test_split_used_for_selection"])
        self.assertGreaterEqual(report["test_targets"], 2)
        self.assertEqual(
            set(report["test_metrics"]),
            {"auc", "auprc", "f1", "precision", "recall"},
        )


if __name__ == "__main__":
    unittest.main()
