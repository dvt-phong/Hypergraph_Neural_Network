"""Small end-to-end check for the CSV-only pipeline."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import preprocess
from features import build_features
from hypergraph import build_hyperedges
from model import build_h0, load_evaluation_data, local_graph, load_train_graph
from train import train, test as evaluate_test


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


class SimplePipelineTest(unittest.TestCase):
    def test_csv_to_model_and_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw, processed = Path(temporary) / "raw", Path(temporary) / "processed"
            raw.mkdir()
            write_csv(raw / "course_info.csv",
                      ("id", "course_id", "start", "end", "course_type", "category"),
                      [(1, "course-a", "2020-01-01", "2020-03-01", 0, "computer"),
                       (2, "course-b", "2020-01-01", "2020-03-01", 0, "")])
            write_csv(raw / "user_info.csv", ("user_id", "gender", "education", "birth"),
                      [(user, "male" if user else "", "High" if user else "", "1990.0")
                       for user in range(12)])
            truths, logs = [], []
            for user in range(12):
                for label in (0, 1):
                    enrollment = user * 2 + label
                    truths.append((enrollment, label))
                    course = "course-a" if user < 6 else "course-b"
                    logs.extend([
                        (enrollment, user, course, f"session-{enrollment}",
                         "play_video", f"video-{label}", "2020-01-02T10:00:00"),
                        (enrollment, user, course, f"session-{enrollment}",
                         "click_info", "", "2020-01-03T10:00:00"),
                    ])
                    if label:
                        logs.append((enrollment, user, course, f"extra-{enrollment}",
                                     "problem_check", "problem-1", "2020-01-04T10:00:00"))
            write_csv(raw / "train_truth.csv", ("enroll_id", "truth"), truths)
            write_csv(raw / "test_truth.csv", ("enroll_id", "truth"), [])
            columns = ("enroll_id", "username", "course_id", "session_id", "action",
                       "object", "time")
            write_csv(raw / "train_log.csv", columns, logs)
            write_csv(raw / "test_log.csv", columns, [])
            summary = preprocess(raw, processed, check_full_counts=False)
            self.assertEqual((summary["nodes"], summary["events_35d"]), (24, 60))
            build_features(processed, buckets=4, seeds=(1,))
            full_x = np.load(processed / "X_seed_1.npy")
            self.assertEqual((full_x[0, 62], full_x[0, 71], full_x[12, 92]),
                             (1, 1, 1))
            build_hyperedges(processed, seed=1, k=2, k_max=3)
            build_h0(processed, seed=1)
            x, h0, labels, families, sizes = load_train_graph(processed, seed=1)
            self.assertEqual(x.shape[1], 60)
            self.assertEqual(h0.shape[0], len(labels))
            self.assertTrue(np.all(sizes >= 2))
            self.assertTrue(np.allclose(x.mean(axis=0), 0, atol=1e-5))
            data = load_evaluation_data(processed, seed=1)
            user_splits = {}
            for node, split in zip(data["nodes"], data["split"]):
                user_splits.setdefault(node["user_id"], set()).add(split)
            self.assertTrue(all(len(splits) == 1 for splits in user_splits.values()))
            target = data["split"].index("validation")
            local_x, local_h, _, _, _ = local_graph(data, target, k=2)
            self.assertEqual(local_x.shape[0], local_h.shape[0])
            self.assertTrue(np.all(local_h.getrow(0).toarray() == 1))
            result = train(seed=1, feature_set="full", output_dir=processed,
                           epochs=1, device_name="cpu",
                           validation_batch_size=2, contrastive_nodes=8, k=2,
                           runs_dir=processed / "runs", reports_dir=processed / "reports",
                           refinement={"sampled_hyperedges": 2, "positive_nodes": 2,
                                       "negative_nodes": 2, "top_r": 2})
            self.assertEqual(result["best_epoch"], 1)
            self.assertGreater(result["history"][0]["scorer_gradient_norm"], 0)
            report = evaluate_test(result["checkpoint"], output_dir=processed,
                                   device_name="cpu", batch_size=2,
                                   reports_dir=processed / "reports")
            self.assertEqual(report["test_targets"], 4)
            self.assertTrue(0 <= report["test_metrics"]["auc"] <= 1)
            one_at_a_time = evaluate_test(result["checkpoint"], output_dir=processed,
                                          device_name="cpu", batch_size=1,
                                          reports_dir=processed / "reports")
            for metric in report["test_metrics"]:
                self.assertAlmostEqual(report["test_metrics"][metric],
                                       one_at_a_time["test_metrics"][metric])
            baseline = train(seed=1, output_dir=processed, epochs=1,
                             device_name="cpu", validation_batch_size=2, hsl=False,
                             runs_dir=processed / "runs", reports_dir=processed / "reports")
            self.assertEqual(baseline["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
