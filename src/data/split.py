# Tạo train/validation/test split deterministic và không trùng user.
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any

import duckdb

from config import (
    DATASET_CONTRACT,
    EXPERIMENT_SEEDS,
    EXPERIMENT_SPLITS,
)
from paths import PROCESSED_DATA_DIR, require_project_path


SPLIT_VERSION = "user-disjoint-v2"
SPLIT_ARTIFACT = "splits.parquet"
SPLIT_ALGORITHM = "stratified weighted round-robin group assignment"
MAX_RELATIVE_DEVIATION = 0.005


# Mục đích: Chuẩn hóa Path để nhúng vào câu SQL DuckDB.
# Đầu vào: Đường dẫn file.
# Đầu ra: Chuỗi đường dẫn tuyệt đối đã escape dấu nháy đơn.
def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


# Mục đích: Ghi một query thành file Parquet hoàn chỉnh.
# Đầu vào: Kết nối DuckDB, câu query và đường dẫn đích.
# Đầu ra: Không trả dữ liệu; tạo hoặc thay thế file đích.
# Lưu ý: Không dùng cache ở bước split để kết quả luôn được kiểm tra lại.
def _write_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    destination: Path,
) -> None:
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    connection.execute(
        f"COPY ({query}) TO '{_sql_path(temporary)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)"
    )
    temporary.replace(destination)


@dataclass(frozen=True)
# Mục đích: Lưu thống kê của một user để gán cả user vào đúng một split.
# Đầu vào: user_id, tổng enrollment và tổng dropout của user.
# Đầu ra: Object bất biến được dùng bởi thuật toán assign_user_groups().
class UserGroup:
    user_id: int
    enrollments: int
    dropouts: int


# Mục đích: Chuyển tỷ lệ thực thành số lượng nguyên mà vẫn giữ đúng tổng.
# Đầu vào: Tổng số phần tử, tuple tỷ lệ và tên các nhóm đích.
# Đầu ra: Dictionary số phần tử nguyên cho từng nhóm.
# Lưu ý: Phần dư được cấp theo largest remainder với tie-break ổn định.
def integer_targets(
    total: int,
    ratios: tuple[float, ...],
    names: tuple[str, ...] = EXPERIMENT_SPLITS,
) -> dict[str, int]:
    if total < 0 or len(names) != len(ratios) or not names:
        raise ValueError("A non-negative total and matching non-empty ratios are required")
    if any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must be non-negative and sum to one")

    exact = [total * ratio for ratio in ratios]
    targets = [math.floor(value) for value in exact]
    remainder = total - sum(targets)
    order = sorted(
        range(len(names)),
        key=lambda index: (-(exact[index] - targets[index]), index),
    )
    for index in order[:remainder]:
        targets[index] += 1
    return dict(zip(names, targets, strict=True))


# Mục đích: Tạo bộ đếm rỗng có cùng cấu trúc cho ba experiment split.
# Đầu vào: Không có.
# Đầu ra: Dictionary users/enrollments/dropouts đều bắt đầu từ 0.
def _new_counts() -> dict[str, dict[str, int]]:
    return {
        split: {"users": 0, "enrollments": 0, "dropouts": 0}
        for split in EXPERIMENT_SPLITS
    }


# Mục đích: Tính mục tiêu users, enrollments và dropouts cho từng split.
# Đầu vào: Danh sách UserGroup và ba tỷ lệ split.
# Đầu ra: Dictionary số lượng mục tiêu theo split và metric.
def _target_counts(
    groups: list[UserGroup], ratios: tuple[float, float, float]
) -> dict[str, dict[str, int]]:
    totals = {
        "users": len(groups),
        "enrollments": sum(group.enrollments for group in groups),
        "dropouts": sum(group.dropouts for group in groups),
    }
    targets = _new_counts()
    for metric, total in totals.items():
        allocated = integer_targets(total, ratios)
        for split in EXPERIMENT_SPLITS:
            targets[split][metric] = allocated[split]
    return targets


# Mục đích: Gán nguyên user vào train/validation/test mà vẫn cân bằng nhãn.
# Đầu vào: Danh sách UserGroup, tỷ lệ split và random seed.
# Đầu ra: Mapping user-to-split, target counts và actual counts.
# Lưu ý: Một user không bao giờ bị tách sang nhiều split; seed quyết định tie-break.
def assign_user_groups(
    groups: list[UserGroup],
    *,
    ratios: tuple[float, float, float] = DATASET_CONTRACT.target_user_ratios,
    seed: int = EXPERIMENT_SEEDS[0],
) -> tuple[dict[int, str], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    if not groups:
        raise ValueError("At least one user group is required")
    if len({group.user_id for group in groups}) != len(groups):
        raise ValueError("Each user_id must appear exactly once")
    if any(
        group.enrollments <= 0
        or group.dropouts < 0
        or group.dropouts > group.enrollments
        for group in groups
    ):
        raise ValueError("Invalid enrollment or dropout count in a user group")

    targets = _target_counts(groups, ratios)
    actual = _new_counts()
    assignments: dict[int, str] = {}
    rng = random.Random(seed)
    ordered = list(groups)
    rng.shuffle(ordered)
    ordered.sort(
        key=lambda group: (
            group.enrollments,
            group.dropouts,
        ),
        reverse=True,
    )

    slot_counts = {split: 0 for split in EXPERIMENT_SPLITS}
    slots: list[str] = []
    for _ in ordered:
        candidates = [
            split
            for split in EXPERIMENT_SPLITS
            if slot_counts[split] < targets[split]["users"]
        ]
        split = min(
            candidates,
            key=lambda candidate: (
                (slot_counts[candidate] + 1) / targets[candidate]["users"],
                EXPERIMENT_SPLITS.index(candidate),
            ),
        )
        slots.append(split)
        slot_counts[split] += 1

    for group, split in zip(ordered, slots, strict=True):
        assignments[group.user_id] = split
        actual[split]["users"] += 1
        actual[split]["enrollments"] += group.enrollments
        actual[split]["dropouts"] += group.dropouts

    if any(actual[split]["users"] != targets[split]["users"] for split in EXPERIMENT_SPLITS):
        raise RuntimeError("The assignment did not meet exact user targets")
    return assignments, targets, actual


# Mục đích: Tổng hợp bảng nodes thành một record thống kê cho mỗi user.
# Đầu vào: Kết nối DuckDB và đường dẫn nodes.parquet.
# Đầu ra: Danh sách UserGroup sắp theo user_id.
def _load_user_groups(
    connection: duckdb.DuckDBPyConnection, nodes_path: Path
) -> list[UserGroup]:
    rows = connection.execute(
        f"""
        SELECT user_id, count(*) AS enrollments,
               sum(CASE WHEN label=1 THEN 1 ELSE 0 END) AS dropouts
        FROM read_parquet('{_sql_path(nodes_path)}')
        GROUP BY user_id ORDER BY user_id
        """
    ).fetchall()
    return [UserGroup(int(row[0]), int(row[1]), int(row[2])) for row in rows]


# Mục đích: Kiểm tra split không overlap và tóm tắt cân bằng theo từng seed.
# Đầu vào: Kết nối, node/split path, target counts và danh sách seed.
# Đầu ra: Hai dictionary gồm summary và invariant cho từng seed.
# Lưu ý: Sai lệch enrollment/dropout phải nhỏ hơn MAX_RELATIVE_DEVIATION.
def _validate_and_summarize(
    connection: duckdb.DuckDBPyConnection,
    nodes_path: Path,
    split_path: Path,
    targets: dict[str, dict[str, int]],
    seeds: tuple[int, ...],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, int]]]:
    split_source = f"read_parquet('{_sql_path(split_path)}')"
    nodes_source = f"read_parquet('{_sql_path(nodes_path)}')"
    seed_values = ", ".join(str(seed) for seed in seeds)
    invariant_rows = connection.execute(
        f"""
        SELECT seed, count(*) AS rows,
               count(DISTINCT node_id) AS node_ids,
               count(DISTINCT enroll_id) AS enrollments,
               count(DISTINCT user_id) AS users,
               count(*) FILTER (
                   WHERE seed IS NULL OR node_id IS NULL OR enroll_id IS NULL
                      OR user_id IS NULL
                      OR source_partition IS NULL OR experiment_split IS NULL
               ) AS missing_required,
               count(*) FILTER (
                   WHERE experiment_split NOT IN ('train', 'validation', 'test')
               ) AS unknown_split
        FROM {split_source} GROUP BY seed
        """
    ).fetchall()
    overlap_rows = connection.execute(
        f"""
        SELECT seed, count(*) FROM (
            SELECT seed, user_id FROM {split_source}
            GROUP BY seed, user_id HAVING count(DISTINCT experiment_split)>1
        ) GROUP BY seed
        """
    ).fetchall()
    overlap_by_seed = {int(row[0]): int(row[1]) for row in overlap_rows}
    row_by_seed = {int(row[0]): row for row in invariant_rows}
    if set(row_by_seed) != set(seeds):
        raise RuntimeError(f"Unexpected seeds in split artifact: {set(row_by_seed)}")

    expected = DATASET_CONTRACT
    invariants_by_seed: dict[str, dict[str, int]] = {}
    for seed in seeds:
        row = row_by_seed[seed]
        invariants = {
            "rows": int(row[1]),
            "distinct_node_ids": int(row[2]),
            "distinct_enrollments": int(row[3]),
            "distinct_users": int(row[4]),
            "missing_required": int(row[5]),
            "unknown_split": int(row[6]),
            "user_overlap": overlap_by_seed.get(seed, 0),
        }
        if invariants != {
            "rows": expected.enrollments,
            "distinct_node_ids": expected.enrollments,
            "distinct_enrollments": expected.enrollments,
            "distinct_users": expected.users,
            "missing_required": 0,
            "unknown_split": 0,
            "user_overlap": 0,
        }:
            raise RuntimeError(f"Split invariant failed for seed {seed}: {invariants}")
        invariants_by_seed[str(seed)] = invariants

    overall = connection.execute(
        f"""
        SELECT count(*), count(*) FILTER (WHERE seed NOT IN ({seed_values}))
        FROM {split_source}
        """
    ).fetchone()
    if int(overall[0]) != expected.enrollments * len(seeds) or int(overall[1]) != 0:
        raise RuntimeError(f"Unexpected combined split rows or seed values: {overall}")

    rows = connection.execute(
        f"""
        SELECT s.seed, s.experiment_split,
               count(DISTINCT s.user_id) AS users,
               count(*) AS enrollments,
               sum(CASE WHEN n.label=1 THEN 1 ELSE 0 END) AS dropouts,
               sum(CASE WHEN n.label=0 THEN 1 ELSE 0 END) AS non_dropouts,
               count(*) FILTER (WHERE s.source_partition='train') AS source_train,
               count(*) FILTER (WHERE s.source_partition='test') AS source_test
        FROM {split_source} s JOIN {nodes_source} n USING(node_id)
        GROUP BY s.seed, s.experiment_split
        """
    ).fetchall()
    row_by_seed_split = {(int(row[0]), row[1]): row for row in rows}

    summary_by_seed: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        summary: dict[str, dict[str, Any]] = {}
        for split in EXPERIMENT_SPLITS:
            row = row_by_seed_split.get((seed, split))
            if row is None:
                raise RuntimeError(f"Missing {split} split for seed {seed}")
            users = int(row[2])
            enrollments = int(row[3])
            dropouts = int(row[4])
            target = targets[split]
            if users != target["users"]:
                raise RuntimeError(
                    f"Unexpected user count in seed {seed}/{split}: {users}"
                )
            enrollment_deviation = enrollments - target["enrollments"]
            dropout_deviation = dropouts - target["dropouts"]
            if (
                abs(enrollment_deviation) / target["enrollments"]
                >= MAX_RELATIVE_DEVIATION
                or abs(dropout_deviation) / target["dropouts"]
                >= MAX_RELATIVE_DEVIATION
            ):
                raise RuntimeError(
                    f"Balance tolerance exceeded in seed {seed}/{split}: "
                    f"enrollments={enrollment_deviation}, "
                    f"dropouts={dropout_deviation}"
                )
            summary[split] = {
                "users": users,
                "target_users": target["users"],
                "enrollments": enrollments,
                "target_enrollments": target["enrollments"],
                "enrollment_deviation": enrollment_deviation,
                "dropouts": dropouts,
                "target_dropouts": target["dropouts"],
                "dropout_deviation": dropout_deviation,
                "non_dropouts": int(row[5]),
                "dropout_rate": dropouts / enrollments,
                "source_train_enrollments": int(row[6]),
                "source_test_enrollments": int(row[7]),
            }
        summary_by_seed[str(seed)] = summary
    return summary_by_seed, invariants_by_seed


# Mục đích: Tạo splits.parquet cho toàn bộ năm seed nghiên cứu.
# Đầu vào: Thư mục processed đã có nodes.parquet.
# Đầu ra: Report mô tả thuật toán, mục tiêu, invariant và thống kê từng seed.
# Lưu ý: Hàm luôn rebuild và tuyệt đối không dùng label validation/test để train model.
def build_splits(
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Any]:
    output_dir = require_project_path(output_dir)
    nodes_path = output_dir / "nodes.parquet"
    if not nodes_path.is_file():
        raise FileNotFoundError(
            f"{nodes_path} does not exist. Run prepare-data before split-data."
        )
    split_path = output_dir / SPLIT_ARTIFACT
    ratios = DATASET_CONTRACT.target_user_ratios
    seeds = EXPERIMENT_SEEDS

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    try:
        groups = _load_user_groups(connection, nodes_path)
        assignment_rows: list[tuple[int, int, str]] = []
        actual_by_seed: dict[int, dict[str, dict[str, int]]] = {}
        targets: dict[str, dict[str, int]] | None = None
        for seed in seeds:
            assignments, seed_targets, actual = assign_user_groups(
                groups, ratios=ratios, seed=seed
            )
            if targets is None:
                targets = seed_targets
            elif targets != seed_targets:
                raise RuntimeError("Split targets changed between seeds")
            actual_by_seed[seed] = actual
            assignment_rows.extend(
                (seed, user_id, split)
                for user_id, split in sorted(assignments.items())
            )
        if targets is None:
            raise RuntimeError("No split targets were generated")
        connection.execute(
            "CREATE TEMP TABLE user_splits "
            "(seed INTEGER NOT NULL, user_id BIGINT NOT NULL, "
            "experiment_split VARCHAR NOT NULL, PRIMARY KEY(seed, user_id))"
        )
        connection.executemany(
            "INSERT INTO user_splits VALUES (?, ?, ?)", assignment_rows
        )
        query = f"""
            SELECT s.seed, n.node_id, n.enroll_id, n.user_id, n.source_partition,
                   s.experiment_split
            FROM read_parquet('{_sql_path(nodes_path)}') n
            JOIN user_splits s USING(user_id)
            ORDER BY s.seed, n.node_id
        """
        _write_parquet(connection, query, split_path)
        summary_by_seed, invariants_by_seed = _validate_and_summarize(
            connection, nodes_path, split_path, targets, seeds
        )
    finally:
        connection.close()

    objective_by_seed = {
        str(seed): sum(
            (
                (actual_by_seed[seed][split][metric] - targets[split][metric])
                / targets[split][metric]
            )
            ** 2
            for split in EXPERIMENT_SPLITS
            for metric in ("users", "enrollments", "dropouts")
        )
        for seed in seeds
    }
    return {
        "split_version": SPLIT_VERSION,
        "dataset": DATASET_CONTRACT.dataset,
        "seeds": list(seeds),
        "algorithm": SPLIT_ALGORITHM,
        "balanced_metrics": ["users", "enrollments", "dropouts"],
        "max_relative_deviation": MAX_RELATIVE_DEVIATION,
        "target_ratios": dict(zip(EXPERIMENT_SPLITS, ratios, strict=True)),
        "objective_by_seed": objective_by_seed,
        "artifact": {
            "path": SPLIT_ARTIFACT,
            "rows": DATASET_CONTRACT.enrollments * len(seeds),
            "size_bytes": split_path.stat().st_size,
        },
        "invariants_by_seed": invariants_by_seed,
        "summary_by_seed": summary_by_seed,
    }
