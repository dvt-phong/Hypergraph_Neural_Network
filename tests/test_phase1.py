from pathlib import Path
import sys
import unittest

import duckdb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.preprocess import PARQUET_ARTIFACTS, validate_source_files
from config import DATASET_CONTRACT, SOURCE_SCHEMAS
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
class PhaseOneSourceTests(unittest.TestCase):
    def test_raw_source_headers(self):
        source = validate_source_files()
        self.assertEqual(set(source), set(SOURCE_SCHEMAS))


@unittest.skipUnless(
    all((PROCESSED_DATA_DIR / name).is_file() for name in PARQUET_ARTIFACTS),
    "Phase 1 Parquet artifacts have not been built",
)
class PhaseOneArtifactTests(unittest.TestCase):
    def test_artifact_counts_and_invariants(self):
        connection = duckdb.connect()
        try:
            paths = {
                name: (PROCESSED_DATA_DIR / name).as_posix()
                for name in PARQUET_ARTIFACTS
            }
            counts = {
                name: connection.execute(
                    f"SELECT count(*) FROM read_parquet('{path}')"
                ).fetchone()[0]
                for name, path in paths.items()
            }
            self.assertEqual(
                counts,
                {
                    "nodes.parquet": DATASET_CONTRACT.enrollments,
                    "events_35d.parquet": DATASET_CONTRACT.retained_events_35d,
                    "users.parquet": DATASET_CONTRACT.users,
                    "courses.parquet": DATASET_CONTRACT.courses,
                },
            )

            node_row = connection.execute(
                f"""
                SELECT count(DISTINCT enroll_id), count(DISTINCT node_id),
                       min(node_id), max(node_id),
                       count(*) FILTER (WHERE label NOT IN (0, 1))
                FROM read_parquet('{paths['nodes.parquet']}')
                """
            ).fetchone()
            self.assertEqual(
                node_row,
                (
                    DATASET_CONTRACT.enrollments,
                    DATASET_CONTRACT.enrollments,
                    0,
                    DATASET_CONTRACT.enrollments - 1,
                    0,
                ),
            )

            event_row = connection.execute(
                f"""
                SELECT min(course_day), max(course_day),
                       count(*) FILTER (WHERE event_time IS NULL)
                FROM read_parquet('{paths['events_35d.parquet']}')
                """
            ).fetchone()
            self.assertEqual(event_row, (0, 34, 0))
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
