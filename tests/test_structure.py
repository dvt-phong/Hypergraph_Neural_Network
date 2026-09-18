from importlib.util import find_spec
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mooc_hgsl.data import inspect_sources
from mooc_hgsl.data.artifacts import FeatureMatrix, IncidenceMatrix, NodeIndex
from mooc_hgsl.data.feature_store import FeatureStore
from mooc_hgsl.data.schema import ACTION_GROUPS, ACTIONS, OBSERVATION_DAYS, feature_columns
from mooc_hgsl.models.hypergraph_structure_learning.model import FLOW_BLOCKS
from mooc_hgsl.models.hypergraph_structure_learning.total_loss import combine_losses


MODEL_PACKAGE = "mooc_hgsl.models.hypergraph_structure_learning"


class ProjectStructureTests(unittest.TestCase):
    def test_each_flowchart_block_has_a_module(self):
        modules = {
            "hypergraph_construction",
            "hgnn",
            "hyperedge_sampling",
            "incident_node_sampling",
            "hypergraph_refinement",
            "contrastive_loss",
            "classifier",
            "total_loss",
            "model",
        }
        for module in modules:
            self.assertIsNotNone(find_spec(f"{MODEL_PACKAGE}.{module}"), module)

    def test_flow_keeps_sampling_steps_separate(self):
        self.assertLess(FLOW_BLOCKS.index("hyperedge_sampling"), FLOW_BLOCKS.index("incident_node_sampling"))
        self.assertLess(FLOW_BLOCKS.index("incident_node_sampling"), FLOW_BLOCKS.index("hypergraph_refinement"))

    def test_total_loss_matches_the_diagram(self):
        loss = combine_losses(2.0, 3.0, 0.1)
        self.assertAlmostEqual(loss.total, 2.3)
        self.assertEqual(loss.bce, 2.0)
        self.assertEqual(loss.contrastive, 3.0)

    def test_large_artifacts_require_sparse_incidence(self):
        nodes = NodeIndex(Path("nodes.parquet"), 10)
        features = FeatureMatrix(Path("features.npy"), 10, 8)
        incidence = IncidenceMatrix(Path("course.npz"), "course", 10, 2, 10)
        nodes.validate()
        features.validate()
        incidence.validate()
        self.assertEqual(incidence.sparse_format, "coo")

    def test_current_data_sources_are_reported_honestly(self):
        statuses = {status.name: status for status in inspect_sources()}
        self.assertIn("full activity", statuses)
        self.assertIn("prediction", statuses)
        if (statuses["full activity"].path / "manifest.json").exists():
            self.assertTrue(statuses["full activity"].ready)

    def test_project_has_no_external_config_directory(self):
        self.assertFalse((ROOT / "configs").exists())

    def test_two_data_branches_share_one_fixed_feature_schema(self):
        # Five summary, four group, 35 daily and 23 action features are expected.
        self.assertEqual(len(feature_columns()), 5 + 4 + 35 + 23)
        self.assertEqual(OBSERVATION_DAYS, 35)
        grouped_actions = tuple(action for group in ACTION_GROUPS.values() for action in group)
        self.assertEqual(set(grouped_actions), set(ACTIONS))
        self.assertEqual(len(grouped_actions), len(ACTIONS))

    def test_shared_feature_store_is_complete_when_built(self):
        store = FeatureStore()
        if store.manifest.exists():
            store.require()
            self.assertEqual(store.root.name, "v1")
            self.assertEqual(store.root.parent.name, "xuetangx_feature_store")


if __name__ == "__main__":
    unittest.main()
