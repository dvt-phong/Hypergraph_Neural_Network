# Transform feature theo train-only statistics và quản lý cache feature artifact.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np

from artifacts import source_signature, sql_path, utc_now, write_json_atomic
from config import (
    COURSE_CATEGORY_VALUES,
    DATASET_CONTRACT,
    EDUCATION_VALUES,
    EXPERIMENT_SEEDS,
    FEATURE_SETS,
    GENDER_VALUES,
    SCHEMA_VERSION,
    base_feature_columns,
    context_feature_columns,
    course_feature_columns,
    feature_columns,
    user_feature_columns,
)
from features.context import build_raw_context
from features.engineering import ABLATION_FEATURES, build_raw_features
from paths import PROCESSED_DATA_DIR, require_project_path


FEATURE_VERSION = "x-feature-groups-v2"
RAW_FEATURES_ARTIFACT = "features_raw.parquet"
RAW_CONTEXT_ARTIFACT = "context_raw.parquet"
X_BASE_ARTIFACT = "X_base.npy"
X_CONTEXT_ARTIFACT = "X_context.npy"
TRANSFORM_ARTIFACT = "feature_transform.joblib"
CONTEXT_TRANSFORM_ARTIFACT = "context_transform.joblib"
FEATURE_MANIFEST = "feature_manifest.json"
TRANSFORM_CHUNK_ROWS = 50_000


# Mục đích: Tính signature có hash cho một nhóm file đầu vào hoặc artifact.
# Đầu vào: Tuple các Path.
# Đầu ra: Dictionary filename-to-signature.
def _signatures(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    return {path.name: source_signature(path, include_hash=True) for path in paths}


# Mục đích: Quyết định feature cache hiện tại có tái sử dụng an toàn hay không.
# Đầu vào: Output directory và signature của các input hiện tại.
# Đầu ra: Manifest nếu cache hợp lệ; None nếu cần rebuild.
# Lưu ý: Kiểm tra cả schema/version, seed, feature names và SHA-256 artifact.
def _feature_cache_hit(
    output_dir: Path,
    input_signatures: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    manifest_path = output_dir / FEATURE_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("feature_version") != FEATURE_VERSION
        or manifest.get("seeds") != list(EXPERIMENT_SEEDS)
        or manifest.get("feature_names") != list(base_feature_columns())
        or manifest.get("context_feature_names")
        != list(context_feature_columns())
        or manifest.get("inputs") != input_signatures
    ):
        return None
    recorded = manifest.get("artifacts", {})
    for name in (
        RAW_FEATURES_ARTIFACT,
        RAW_CONTEXT_ARTIFACT,
        X_BASE_ARTIFACT,
        X_CONTEXT_ARTIFACT,
        TRANSFORM_ARTIFACT,
        CONTEXT_TRANSFORM_ARTIFACT,
    ):
        path = output_dir / name
        if not path.is_file() or recorded.get(name) is None:
            return None
        if source_signature(path, include_hash=True) != recorded[name]:
            return None
    return manifest


# Mục đích: Đọc feature hành vi thô thành matrix NumPy và áp dụng log1p.
# Đầu vào: Kết nối DuckDB và features_raw.parquet.
# Đầu ra: Matrix float32 [225642, 60] đã log1p.
# Lưu ý: Chưa standardize; bước đó phải fit riêng trên train của từng seed.
def _load_raw_matrix(
    connection: duckdb.DuckDBPyConnection,
    features_path: Path,
) -> np.ndarray:
    columns = base_feature_columns()
    selected = ", ".join(columns)
    arrays = connection.execute(
        f"SELECT {selected} FROM read_parquet('{sql_path(features_path)}') "
        "ORDER BY node_id"
    ).fetchnumpy()
    matrix = np.empty(
        (DATASET_CONTRACT.enrollments, len(columns)), dtype=np.float32
    )
    for index, column in enumerate(columns):
        matrix[:, index] = arrays[column]
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise RuntimeError("Raw X_base contains non-finite or negative values")
    np.log1p(matrix, out=matrix)
    return matrix


# Mục đích: Lấy global node_id thuộc train split của một seed.
# Đầu vào: Kết nối, splits.parquet và seed.
# Đầu ra: Vector int64 node_id đã sort.
# Lưu ý: Các ID này là nguồn duy nhất để fit transform, tránh leakage.
def _train_node_ids(
    connection: duckdb.DuckDBPyConnection,
    splits_path: Path,
    seed: int,
) -> np.ndarray:
    result = connection.execute(
        f"""
        SELECT node_id FROM read_parquet('{sql_path(splits_path)}')
        WHERE seed=? AND experiment_split='train' ORDER BY node_id
        """,
        [seed],
    ).fetchnumpy()
    return result["node_id"].astype(np.int64, copy=False)


# Mục đích: Đọc năm cột context thô thành các NumPy array đồng bộ node order.
# Đầu vào: Kết nối DuckDB và context_raw.parquet.
# Đầu ra: Dictionary array categorical và numeric theo tên cột.
# Lưu ý: NULL numeric được chuyển thành NaN để impute ở bước train-only.
def _load_context_arrays(
    connection: duckdb.DuckDBPyConnection,
    context_path: Path,
) -> dict[str, np.ndarray]:
    arrays = connection.execute(
        f"""SELECT gender, education, age_at_course_start, category,
                   course_duration_days
            FROM read_parquet('{sql_path(context_path)}') ORDER BY node_id"""
    ).fetchnumpy()
    output: dict[str, np.ndarray] = {}
    for name in ("gender", "education", "category"):
        values = np.asarray(np.ma.asarray(arrays[name]).filled(None), dtype=object)
        if values.shape != (DATASET_CONTRACT.enrollments,):
            raise RuntimeError(f"Unexpected context column shape for {name}")
        output[name] = values
    for name in ("age_at_course_start", "course_duration_days"):
        values = np.ma.asarray(arrays[name], dtype=np.float64)
        output[name] = np.asarray(values.filled(np.nan), dtype=np.float64)
    return output


# Mục đích: One-hot một cột category vào đúng block của output matrix.
# Đầu vào: Matrix output, giá trị thô, vocabulary cố định và vị trí bắt đầu.
# Đầu ra: Không trả riêng; cập nhật matrix output tại chỗ.
# Lưu ý: Có hai cột cuối cho missing và category ngoài vocabulary.
def _encode_one_hot(
    output: np.ndarray,
    values: np.ndarray,
    vocabulary: tuple[str, ...],
    start: int,
) -> None:
    missing = np.fromiter(
        (value is None for value in values), dtype=bool, count=values.size
    )
    known = np.zeros(values.size, dtype=bool)
    for offset, value in enumerate(vocabulary):
        matches = values == value
        output[matches, start + offset] = 1.0
        known |= matches
    output[missing, start + len(vocabulary)] = 1.0
    output[~missing & ~known, start + len(vocabulary) + 1] = 1.0


# Mục đích: Impute và standardize một feature numeric bằng train statistics.
# Đầu vào: Toàn bộ values và node_id thuộc train.
# Đầu ra: Giá trị chuẩn hóa, cờ missing và dictionary tham số transform.
# Lưu ý: Median/mean/std chỉ được tính từ train; scale 0 được thay bằng 1.
def _standardize_context_numeric(
    values: np.ndarray,
    train_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    missing = ~np.isfinite(values)
    train_observed = values[train_ids][~missing[train_ids]]
    if train_observed.size == 0:
        raise RuntimeError("A context numeric feature has no observed train values")
    median = float(np.median(train_observed))
    imputed = np.where(missing, median, values)
    train = imputed[train_ids]
    mean = float(np.mean(train, dtype=np.float64))
    scale = float(np.std(train, dtype=np.float64))
    zero_variance = scale == 0.0
    if zero_variance:
        scale = 1.0
    transformed = ((imputed - mean) / scale).astype(np.float32)
    parameters = {
        "median": median,
        "mean": mean,
        "scale": scale,
        "zero_variance": zero_variance,
    }
    return transformed, missing.astype(np.float32), parameters


# Mục đích: Tạo X_context cho toàn bộ seed bằng vocabulary và train statistics.
# Đầu vào: Kết nối, context path, split path và file .npy đích.
# Đầu ra: Transform metadata và audit theo seed.
# Lưu ý: Array layout là [seed_index, node_id, context_feature_index].
def _write_x_context(
    connection: duckdb.DuckDBPyConnection,
    context_path: Path,
    splits_path: Path,
    destination: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = _load_context_arrays(connection, context_path)
    dimension = len(context_feature_columns())
    categorical = np.zeros(
        (DATASET_CONTRACT.enrollments, dimension), dtype=np.float32
    )
    gender_start = 0
    education_start = len(GENDER_VALUES) + 2
    age_index = education_start + len(EDUCATION_VALUES) + 2
    category_start = len(user_feature_columns())
    duration_index = category_start + len(COURSE_CATEGORY_VALUES) + 2
    _encode_one_hot(categorical, arrays["gender"], GENDER_VALUES, gender_start)
    _encode_one_hot(
        categorical, arrays["education"], EDUCATION_VALUES, education_start
    )
    _encode_one_hot(
        categorical, arrays["category"], COURSE_CATEGORY_VALUES, category_start
    )

    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    shape = (
        len(EXPERIMENT_SEEDS),
        DATASET_CONTRACT.enrollments,
        dimension,
    )
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=shape
    )
    transform_by_seed: dict[str, Any] = {}
    audit_by_seed: dict[str, Any] = {}
    try:
        for seed_index, seed in enumerate(EXPERIMENT_SEEDS):
            train_ids = _train_node_ids(connection, splits_path, seed)
            age, age_missing, age_parameters = _standardize_context_numeric(
                arrays["age_at_course_start"], train_ids
            )
            duration, duration_missing, duration_parameters = (
                _standardize_context_numeric(
                    arrays["course_duration_days"], train_ids
                )
            )
            output[seed_index] = categorical
            output[seed_index, :, age_index] = age
            output[seed_index, :, age_index + 1] = age_missing
            output[seed_index, :, duration_index] = duration
            output[seed_index, :, duration_index + 1] = duration_missing
            transform_by_seed[str(seed)] = {
                "train_nodes": int(train_ids.size),
                "age_at_course_start": age_parameters,
                "course_duration_days": duration_parameters,
            }
            audit_by_seed[str(seed)] = {
                "train_nodes": int(train_ids.size),
                "max_abs_numeric_train_mean": float(
                    max(
                        abs(np.mean(age[train_ids], dtype=np.float64)),
                        abs(np.mean(duration[train_ids], dtype=np.float64)),
                    )
                ),
                "max_abs_numeric_train_std_error": float(
                    max(
                        abs(np.std(age[train_ids], dtype=np.float64) - 1.0),
                        abs(
                            np.std(duration[train_ids], dtype=np.float64) - 1.0
                        ),
                    )
                ),
            }
        output.flush()
    finally:
        del output
    temporary.replace(destination)
    transform = {
        "feature_version": FEATURE_VERSION,
        "feature_names": list(context_feature_columns()),
        "operation": (
            "fixed-vocabulary one-hot; train-median imputation and "
            "train-only standardization for numeric context"
        ),
        "vocabularies": {
            "gender": list(GENDER_VALUES),
            "education": list(EDUCATION_VALUES),
            "category": list(COURSE_CATEGORY_VALUES),
        },
        "seed_order": list(EXPERIMENT_SEEDS),
        "by_seed": transform_by_seed,
    }
    return transform, audit_by_seed


# Mục đích: Standardize X_base riêng cho từng seed và ghi memory-mapped .npy.
# Đầu vào: Kết nối, matrix đã log1p, split path và file đích.
# Đầu ra: Transform metadata và audit mean/std theo seed.
# Lưu ý: Chia chunk khi ghi để không tạo thêm bản sao toàn bộ matrix trong RAM.
def _write_x_base(
    connection: duckdb.DuckDBPyConnection,
    logged_matrix: np.ndarray,
    splits_path: Path,
    destination: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    shape = (
        len(EXPERIMENT_SEEDS),
        DATASET_CONTRACT.enrollments,
        len(base_feature_columns()),
    )
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=shape
    )
    transform_by_seed: dict[str, Any] = {}
    audit_by_seed: dict[str, Any] = {}
    try:
        for seed_index, seed in enumerate(EXPERIMENT_SEEDS):
            train_ids = _train_node_ids(connection, splits_path, seed)
            train_matrix = logged_matrix[train_ids]
            mean = np.mean(train_matrix, axis=0, dtype=np.float64)
            standard_deviation = np.std(train_matrix, axis=0, dtype=np.float64)
            zero_variance = standard_deviation == 0
            scale = standard_deviation.copy()
            scale[zero_variance] = 1.0
            mean32 = mean.astype(np.float32)
            scale32 = scale.astype(np.float32)

            for start in range(0, logged_matrix.shape[0], TRANSFORM_CHUNK_ROWS):
                stop = min(start + TRANSFORM_CHUNK_ROWS, logged_matrix.shape[0])
                output[seed_index, start:stop] = (
                    logged_matrix[start:stop] - mean32
                ) / scale32

            transformed_train = (train_matrix - mean32) / scale32
            variable = ~zero_variance
            train_mean = np.mean(transformed_train, axis=0, dtype=np.float64)
            train_std = np.std(transformed_train, axis=0, dtype=np.float64)
            max_mean_error = (
                float(np.max(np.abs(train_mean[variable])))
                if np.any(variable)
                else 0.0
            )
            max_std_error = (
                float(np.max(np.abs(train_std[variable] - 1.0)))
                if np.any(variable)
                else 0.0
            )
            zero_variance_features = [
                name
                for name, is_zero in zip(
                    base_feature_columns(), zero_variance, strict=True
                )
                if is_zero
            ]
            transform_by_seed[str(seed)] = {
                "mean": mean32,
                "scale": scale32,
                "train_nodes": int(train_ids.size),
                "zero_variance_features": zero_variance_features,
            }
            audit_by_seed[str(seed)] = {
                "train_nodes": int(train_ids.size),
                "max_abs_train_mean": max_mean_error,
                "max_abs_train_std_error": max_std_error,
                "zero_variance_features": zero_variance_features,
            }
            del train_matrix, transformed_train
        output.flush()
    finally:
        del output
    temporary.replace(destination)
    transform = {
        "feature_version": FEATURE_VERSION,
        "feature_names": list(base_feature_columns()),
        "operation": "log1p followed by train-only standardization",
        "seed_order": list(EXPERIMENT_SEEDS),
        "by_seed": transform_by_seed,
    }
    return transform, audit_by_seed


# Mục đích: Ghi dictionary tham số transform bằng joblib theo cách atomic.
# Đầu vào: File đích và dictionary transform.
# Đầu ra: Không trả dữ liệu; tạo file joblib nén.
def _write_transform(path: Path, transform: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    joblib.dump(transform, temporary, compress=3)
    temporary.replace(path)


# Mục đích: Chạy trọn Phase 3 để tạo behavioral/context features cho năm seed.
# Đầu vào: Output directory và cờ force rebuild.
# Đầu ra: Feature manifest mô tả schema, transform, audit và artifact signatures.
# Lưu ý: Mọi scaler/imputer chỉ fit trên train của seed tương ứng.
def build_features(
    output_dir: Path = PROCESSED_DATA_DIR,
    *,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = require_project_path(output_dir)
    nodes_path = output_dir / "nodes.parquet"
    events_path = output_dir / "events_35d.parquet"
    users_path = output_dir / "users.parquet"
    courses_path = output_dir / "courses.parquet"
    splits_path = output_dir / "splits.parquet"
    input_paths = (
        nodes_path,
        events_path,
        users_path,
        courses_path,
        splits_path,
    )
    missing = [path.name for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing preprocessing artifacts: "
            f"{missing}. Run prepare-data and split-data first."
        )
    input_signatures = _signatures(input_paths)
    if not force and (
        cached := _feature_cache_hit(output_dir, input_signatures)
    ) is not None:
        return {**cached, "cache_hit": True}

    raw_features_path = output_dir / RAW_FEATURES_ARTIFACT
    raw_context_path = output_dir / RAW_CONTEXT_ARTIFACT
    x_base_path = output_dir / X_BASE_ARTIFACT
    x_context_path = output_dir / X_CONTEXT_ARTIFACT
    transform_path = output_dir / TRANSFORM_ARTIFACT
    context_transform_path = output_dir / CONTEXT_TRANSFORM_ARTIFACT
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    try:
        raw_audit = build_raw_features(
            connection, nodes_path, events_path, raw_features_path
        )
        context_audit = build_raw_context(
            connection,
            nodes_path,
            users_path,
            courses_path,
            raw_context_path,
        )
        logged_matrix = _load_raw_matrix(connection, raw_features_path)
        transform, transform_audit = _write_x_base(
            connection, logged_matrix, splits_path, x_base_path
        )
        del logged_matrix
        context_transform, context_transform_audit = _write_x_context(
            connection, raw_context_path, splits_path, x_context_path
        )
    finally:
        connection.close()
    _write_transform(transform_path, transform)
    _write_transform(context_transform_path, context_transform)

    if _signatures(input_paths) != input_signatures:
        raise RuntimeError("A Phase 1/2 artifact changed while features were being built")
    artifacts = _signatures(
        (
            raw_features_path,
            raw_context_path,
            x_base_path,
            x_context_path,
            transform_path,
            context_transform_path,
        )
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "dataset": DATASET_CONTRACT.dataset,
        "generated_utc": utc_now(),
        "cache_hit": False,
        "seeds": list(EXPERIMENT_SEEDS),
        "feature_names": list(base_feature_columns()),
        "context_feature_names": list(context_feature_columns()),
        "user_feature_names": list(user_feature_columns()),
        "course_feature_names": list(course_feature_columns()),
        "feature_sets": {
            name: {
                "dimension": len(feature_columns(name)),
                "feature_names": list(feature_columns(name)),
            }
            for name in FEATURE_SETS
        },
        "ablation_features": list(ABLATION_FEATURES),
        "transform": "log1p followed by train-only standardization",
        "array_layout": "X_base[seed_index, node_id, feature_index]",
        "array_shape": [
            len(EXPERIMENT_SEEDS),
            DATASET_CONTRACT.enrollments,
            len(base_feature_columns()),
        ],
        "array_dtype": "float32",
        "context_array_layout": (
            "X_context[seed_index, node_id, context_feature_index]"
        ),
        "context_array_shape": [
            len(EXPERIMENT_SEEDS),
            DATASET_CONTRACT.enrollments,
            len(context_feature_columns()),
        ],
        "context_array_dtype": "float32",
        "context_blocks": {
            "user": [0, len(user_feature_columns())],
            "course": [
                len(user_feature_columns()),
                len(context_feature_columns()),
            ],
        },
        "inputs": input_signatures,
        "artifacts": artifacts,
        "raw_feature_audit": raw_audit,
        "raw_context_audit": context_audit,
        "transform_audit_by_seed": transform_audit,
        "context_transform_audit_by_seed": context_transform_audit,
    }
    write_json_atomic(output_dir / FEATURE_MANIFEST, manifest)
    return manifest
