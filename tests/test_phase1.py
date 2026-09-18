import json
from pathlib import Path
import sys
import unittest

import duckdb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.audit import SOURCE_SCHEMAS, validate_source_files
from data.preprocess import PARQUET_ARTIFACTS, prepare_dataset
from data.schema import DATASET_CONTRACT, SCHEMA_VERSION
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


class PhaseOneContractTests(unittest.TestCase):
    def test_source_contract_contains_only_labeled_xuetangx_files(self):
        self.assertEqual(
            set(SOURCE_SCHEMAS),
            {
                "train_log.csv",
                "test_log.csv",
                "train_truth.csv",
                "test_truth.csv",
                "user_info.csv",
                "course_info.csv",
            },
        )

    def test_phase_one_artifact_names_are_flat(self):
        self.assertEqual(
            PARQUET_ARTIFACTS,
            (
                "nodes.parquet",
                "events_35d.parquet",
                "users.parquet",
                "courses.parquet",
            ),
        )


@unittest.skipUnless(RAW_DATA_DIR.is_dir(), "raw XuetangX data is not available")
class PhaseOneIntegrationTests(unittest.TestCase):
    def test_raw_source_headers(self):
        source = validate_source_files()
        self.assertEqual(set(source), set(SOURCE_SCHEMAS))

    @unittest.skipUnless(
        (PROCESSED_DATA_DIR / "manifest.json").is_file(),
        "Phase 1 artifacts have not been built",
    )
    def test_artifact_counts_and_invariants(self):
        manifest = json.loads(
            (PROCESSED_DATA_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        expected = {
            "nodes.parquet": DATASET_CONTRACT.enrollments,
            "events_35d.parquet": DATASET_CONTRACT.retained_events_35d,
            "users.parquet": DATASET_CONTRACT.users,
            "courses.parquet": DATASET_CONTRACT.courses,
        }
        actual = {item["path"]: item["rows"] for item in manifest["artifacts"]}
        self.assertEqual(actual, expected)

        audit = json.loads(
            (PROCESSED_DATA_DIR / "audit.json").read_text(encoding="utf-8")
        )
        self.assertTrue(all(value == 0 for value in audit["enrollment_relations"].values()))
        self.assertEqual(audit["user_profiles"]["selected_users"], DATASET_CONTRACT.users)
        self.assertEqual(audit["user_profiles"]["missing_metadata"], 0)
        self.assertEqual(audit["course_duration"]["used_courses"], DATASET_CONTRACT.courses)
        self.assertEqual(audit["course_duration"]["missing_metadata"], 0)

        connection = duckdb.connect()
        try:
            nodes = (PROCESSED_DATA_DIR / "nodes.parquet").as_posix()
            events = (PROCESSED_DATA_DIR / "events_35d.parquet").as_posix()
            node_row = connection.execute(
                f"""
                SELECT count(*), count(DISTINCT enroll_id), count(DISTINCT node_id),
                       count(*) FILTER (WHERE label NOT IN (0, 1))
                FROM read_parquet('{nodes}')
                """
            ).fetchone()
            self.assertEqual(
                node_row,
                (
                    DATASET_CONTRACT.enrollments,
                    DATASET_CONTRACT.enrollments,
                    DATASET_CONTRACT.enrollments,
                    0,
                ),
            )
            event_row = connection.execute(
                f"""
                SELECT count(*), min(course_day), max(course_day),
                       count(*) FILTER (WHERE event_time IS NULL)
                FROM read_parquet('{events}')
                """
            ).fetchone()
            self.assertEqual(
                event_row,
                (DATASET_CONTRACT.retained_events_35d, 0, 34, 0),
            )
        finally:
            connection.close()

    @unittest.skipUnless(
        (PROCESSED_DATA_DIR / "manifest.json").is_file(),
        "Phase 1 artifacts have not been built",
    )
    def test_second_prepare_uses_cache(self):
        manifest = prepare_dataset()
        self.assertTrue(manifest["cache_hit"])


if __name__ == "__main__":
    unittest.main()
