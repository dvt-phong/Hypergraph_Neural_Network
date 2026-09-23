# Check the research pipeline and its methodological invariants.

import ast
import csv
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import download as download_module
from features import (
    BEHAVIOR_FEATURE_COUNT,
    COURSE_FEATURE_START,
    TOTAL_FEATURE_COUNT,
    USER_FEATURE_START,
    build_features,
)
from hypergraph import (
    build_hypergraph,
    build_local_graph,
    load_evaluation_data,
    load_train_graph,
)
from model import HGSLModel
from preprocess import preprocess, read_csv, split_nodes
from train import test as evaluate_test
from train import train


class SourceReadabilityTest(unittest.TestCase):
    # Require explicit control flow and a line comment above every source function.
    def test_source_uses_explicit_readable_constructs(self):
        source_dir = Path(__file__).resolve().parents[1] / "src"
        compact_node_types = (
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.IfExp,
            ast.Lambda,
        )

        problems = []
        for source_path in sorted(source_dir.glob("*.py")):
            source_text = source_path.read_text(encoding="utf-8")
            source_lines = source_text.splitlines()
            syntax_tree = ast.parse(source_text)

            if '\"\"\"' in source_text or "'''" in source_text:
                problems.append(f"{source_path.name}: triple-quoted text")

            for syntax_node in ast.walk(syntax_tree):
                if isinstance(syntax_node, compact_node_types):
                    problems.append(
                        f"{source_path.name}:{syntax_node.lineno}: "
                        f"{type(syntax_node).__name__}"
                    )
                if not isinstance(syntax_node, ast.FunctionDef):
                    continue

                previous_line_index = syntax_node.lineno - 2
                while (
                    previous_line_index >= 0
                    and not source_lines[previous_line_index].strip()
                ):
                    previous_line_index -= 1
                has_comment = (
                    previous_line_index >= 0
                    and source_lines[previous_line_index].lstrip().startswith("#")
                )
                if not has_comment:
                    problems.append(
                        f"{source_path.name}:{syntax_node.lineno}: "
                        f"missing comment above {syntax_node.name}"
                    )

        self.assertEqual(problems, [])


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


def csv_bytes(columns, rows):
    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(columns)
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def add_archive_csv(archive, name, columns, rows):
    content = csv_bytes(columns, rows)
    member = tarfile.TarInfo(name=f"prediction_data/{name}")
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def make_raw_fixture(raw):
    raw.mkdir()
    write_csv(
        raw / "course_info.csv",
        ("id", "course_id", "start", "end", "course_type", "category"),
        [
            (1, "course-a", "2020-01-01", "2020-03-01", 0, "computer"),
            (2, "course-b", "2020-01-01", "2020-03-01", 0, ""),
        ],
    )
    user_rows = []
    for user_id in range(12):
        gender = "male" if user_id else ""
        education = "High" if user_id else ""
        user_rows.append((user_id, gender, education, "1990.0"))
    write_csv(
        raw / "user_info.csv",
        ("user_id", "gender", "education", "birth"),
        user_rows,
    )

    truths = []
    logs = []
    for user_id in range(12):
        for label in (0, 1):
            enrollment = user_id * 2 + label
            truths.append((enrollment, label))
            course_id = "course-a" if user_id < 6 else "course-b"
            logs.append((
                enrollment,
                user_id,
                course_id,
                f"session-{enrollment}",
                "play_video",
                f"video-{label}",
                "2020-01-02T10:00:00",
            ))
            logs.append((
                enrollment,
                user_id,
                course_id,
                f"session-{enrollment}",
                "click_info",
                "",
                "2020-01-03T10:00:00",
            ))
            if label:
                logs.append((
                    enrollment,
                    user_id,
                    course_id,
                    f"extra-{enrollment}",
                    "problem_check",
                    "problem-1",
                    "2020-01-04T10:00:00",
                ))

    log_columns = (
        "enroll_id", "username", "course_id", "session_id", "action", "object", "time"
    )
    with tarfile.open(raw / "prediction_data.tar.gz", "w:gz") as archive:
        add_archive_csv(
            archive,
            "train_truth.csv",
            ("enroll_id", "truth"),
            truths,
        )
        add_archive_csv(
            archive,
            "test_truth.csv",
            ("enroll_id", "truth"),
            [],
        )
        add_archive_csv(archive, "train_log.csv", log_columns, logs)
        add_archive_csv(archive, "test_log.csv", log_columns, [])


class DownloadTest(unittest.TestCase):
    def test_download_only_fetches_missing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)

            def fake_download(url, destination):
                Path(destination).write_bytes(url.encode("utf-8"))

            with mock.patch.object(
                download_module.urllib.request,
                "urlretrieve",
                side_effect=fake_download,
            ) as retrieve:
                download_module.download(raw)
                self.assertEqual(retrieve.call_count, 3)
                download_module.download(raw)
                self.assertEqual(retrieve.call_count, 3)


class SparseGradientTest(unittest.TestCase):
    def test_native_sparse_mm_reaches_membership_scorer(self):
        torch.manual_seed(7)
        h0 = sparse.csr_matrix(np.asarray([
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ], dtype=np.float32))
        x = torch.randn(6, 4)
        model = HGSLModel(input_dim=4, hidden_dim=8, dropout=0.0)
        output = model(
            x,
            h0,
            np.asarray(["course", "object"]),
            np.asarray([3, 4]),
            np.random.default_rng(7),
            refinement={
                "sampled_hyperedges": 2,
                "positive_nodes": 2,
                "negative_nodes": 2,
                "top_r": 2,
            },
        )
        output["z_star"].square().sum().backward()

        parameters = (
            model.node_projection.weight,
            model.edge_projection.weight,
            model.membership_bias,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.norm()), 0)


class SimplePipelineTest(unittest.TestCase):
    def test_archive_to_model_and_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"
            processed = Path(temporary) / "processed"
            make_raw_fixture(raw)

            summary = preprocess(raw, processed)
            self.assertEqual((summary["nodes"], summary["events_35d"]), (24, 60))
            self.assertFalse((raw / "train_log.csv").exists())
            self.assertFalse((raw / "train_truth.csv").exists())

            nodes = list(read_csv(processed / "nodes.csv"))
            split = split_nodes(nodes, 1)
            self.assertEqual(split, split_nodes(nodes, 1))
            self.assertNotEqual(split, split_nodes(nodes, 11))
            user_splits = {}
            for node, split_name in zip(nodes, split):
                user_splits.setdefault(node["user_id"], set()).add(split_name)
            self.assertTrue(all(len(names) == 1 for names in user_splits.values()))

            build_features(processed, seed=1, buckets=4)
            self.assertTrue((processed / "feature_base.npz").exists())
            self.assertTrue((processed / "node_objects.csv.gz").exists())
            full_x = np.load(processed / "X_seed_1.npy")
            self.assertEqual(full_x.shape, (24, TOTAL_FEATURE_COUNT))
            self.assertEqual(BEHAVIOR_FEATURE_COUNT, 60)
            self.assertEqual(USER_FEATURE_START, 60)
            self.assertEqual(COURSE_FEATURE_START, 75)
            build_features(processed, seed=11, buckets=4)
            self.assertTrue((processed / "X_seed_11.npy").exists())

            graph_report = build_hypergraph(processed, seed=1, k=2, k_max=3)
            self.assertEqual(set(graph_report["families"]), {
                "course", "object", "behavioral"
            })
            x, h0, labels, families, sizes = load_train_graph(processed, seed=1)
            self.assertEqual(x.shape[1], BEHAVIOR_FEATURE_COUNT)
            self.assertEqual(h0.shape[0], len(labels))
            self.assertTrue(np.all(sizes >= 2))
            self.assertTrue(np.allclose(x.mean(axis=0), 0, atol=1e-5))

            data = load_evaluation_data(processed, seed=1)
            target = data["split"].index("validation")
            local_x, local_h, _, _, _ = build_local_graph(data, target)
            self.assertEqual(local_x.shape[0], local_h.shape[0])
            self.assertTrue(np.all(local_h.getrow(0).toarray() == 1))
            local_ids = set()
            for course_nodes in data["course_references"].values():
                local_ids.update(course_nodes)
            self.assertTrue(all(data["split"][node_id] == "train" for node_id in local_ids))

            result = train(
                seed=1,
                feature_set="full",
                output_dir=processed,
                epochs=1,
                device_name="cpu",
                validation_batch_size=2,
                contrastive_nodes=8,
                runs_dir=processed / "runs",
                reports_dir=processed / "reports",
                refinement={
                    "sampled_hyperedges": 2,
                    "positive_nodes": 2,
                    "negative_nodes": 2,
                    "top_r": 2,
                },
            )
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(
                Path(result["checkpoint"]).name,
                "simple_hgsl_full_seed_1.pt",
            )

            report = evaluate_test(
                result["checkpoint"],
                output_dir=processed,
                device_name="cpu",
                batch_size=2,
                reports_dir=processed / "reports",
            )
            self.assertEqual(report["test_targets"], 4)
            self.assertTrue(0 <= report["test_metrics"]["auc"] <= 1)

            one_at_a_time = evaluate_test(
                result["checkpoint"],
                output_dir=processed,
                device_name="cpu",
                batch_size=1,
                reports_dir=processed / "reports",
            )
            for metric_name, metric_value in report["test_metrics"].items():
                self.assertAlmostEqual(
                    metric_value,
                    one_at_a_time["test_metrics"][metric_name],
                )

            baseline = train(
                seed=1,
                output_dir=processed,
                epochs=1,
                device_name="cpu",
                validation_batch_size=2,
                hsl=False,
                runs_dir=processed / "runs",
                reports_dir=processed / "reports",
            )
            self.assertEqual(baseline["best_epoch"], 1)

            (raw / "prediction_data.tar.gz").unlink()
            skipped = preprocess(raw, processed)
            self.assertTrue(skipped["skipped"])


if __name__ == "__main__":
    unittest.main()
