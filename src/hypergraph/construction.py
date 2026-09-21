# Điều phối Phase 4-5: tạo hyperedge candidates và sparse initial hypergraph H0.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from scipy import sparse

from artifacts import (
    copy_parquet_atomic,
    source_signature,
    sql_path,
    utc_now,
    write_json_atomic,
)
from config import (
    DATASET_CONTRACT,
    EXPERIMENT_SEEDS,
    SCHEMA_VERSION,
    base_feature_columns,
)
from features.transform import build_features
from hypergraph.audit import audit_behavioral, audit_structural
from hypergraph.behavioral import (
    DEFAULT_K,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    K_CANDIDATES,
    K_MAX,
    build_seed_neighbors,
    write_neighbors_atomic,
)
from hypergraph.course import course_membership_query
from hypergraph.object import object_membership_query
from paths import PROCESSED_DATA_DIR, require_project_path


HYPEREDGE_VERSION = "hyperedge-families-v1"
STRUCTURAL_ARTIFACT = "structural_memberships.parquet"
BEHAVIORAL_ARTIFACT = "behavioral_neighbors.npz"
AUDIT_ARTIFACT = "hyperedge_audit.json"
MANIFEST_ARTIFACT = "hyperedge_manifest.json"
GRAPH_VERSION = "initial-hypergraph-v1"
TRAIN_NODE_INDEX_ARTIFACT = "train_node_index.parquet"
HYPEREDGE_METADATA_ARTIFACT = "hyperedges.parquet"
VALIDATION_MEMBERSHIPS_ARTIFACT = "validation_memberships.parquet"
TEST_MEMBERSHIPS_ARTIFACT = "test_memberships.parquet"
GRAPH_AUDIT_ARTIFACT = "hypergraph_audit.json"
GRAPH_MANIFEST_ARTIFACT = "hypergraph_manifest.json"


# Mục đích: Tính SHA-256 signature cho nhóm input/artifact của hypergraph.
# Đầu vào: Tuple đường dẫn file.
# Đầu ra: Dictionary filename-to-signature.
def _signatures(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    return {path.name: source_signature(path, include_hash=True) for path in paths}


# Mục đích: Kiểm tra cache Phase 4 còn khớp source và cấu hình hiện tại.
# Đầu vào: Output directory và input signatures mới tính.
# Đầu ra: Manifest hợp lệ hoặc None nếu cần rebuild.
# Lưu ý: Kiểm tra hash của structural, behavioral và audit artifacts.
def _cache_hit(
    output_dir: Path, inputs: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    path = output_dir / MANIFEST_ARTIFACT
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("hyperedge_version") != HYPEREDGE_VERSION
        or manifest.get("seeds") != list(EXPERIMENT_SEEDS)
        or manifest.get("k_candidates") != list(K_CANDIDATES)
        or manifest.get("inputs") != inputs
    ):
        return None
    artifacts = manifest.get("artifacts", {})
    for name in (STRUCTURAL_ARTIFACT, BEHAVIORAL_ARTIFACT, AUDIT_ARTIFACT):
        artifact = output_dir / name
        if not artifact.is_file() or artifacts.get(name) != source_signature(
            artifact, include_hash=True
        ):
            return None
    return manifest


# Mục đích: Ghép Course và Object candidate memberships vào một Parquet.
# Đầu vào: Kết nối, nodes/events paths và file đích.
# Đầu ra: Không trả dữ liệu; ghi structural_memberships.parquet.
# Lưu ý: Chưa lọc theo split ở bước global candidate này.
def _build_structural(
    connection: duckdb.DuckDBPyConnection,
    nodes_path: Path,
    events_path: Path,
    destination: Path,
) -> None:
    query = f"""
        SELECT * FROM ({course_membership_query(nodes_path)})
        UNION ALL
        SELECT * FROM ({object_membership_query(events_path)})
        ORDER BY family, course_id, object_type, object_id, node_id
    """
    copy_parquet_atomic(connection, query, destination)


# Mục đích: Đọc global node_id thuộc train của một seed.
# Đầu vào: Kết nối, splits.parquet và seed.
# Đầu ra: Vector int64 node IDs đã sort.
def _train_ids(
    connection: duckdb.DuckDBPyConnection, splits_path: Path, seed: int
) -> np.ndarray:
    result = connection.execute(
        f"""SELECT node_id FROM read_parquet('{sql_path(splits_path)}')
            WHERE seed=? AND experiment_split='train' ORDER BY node_id""",
        [seed],
    ).fetchnumpy()
    return result["node_id"].astype(np.int64, copy=False)


# Mục đích: Tạo toàn bộ candidate Course/Object/Behavioral dùng lại ở Phase 5.
# Đầu vào: Output directory và cờ force rebuild.
# Đầu ra: Manifest chứa audit, k candidates và artifact signatures.
# Lưu ý: Behavioral kNN chỉ fit trên train X_base và không đọc label.
def build_hyperedges(
    output_dir: Path = PROCESSED_DATA_DIR,
    *,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = require_project_path(output_dir)
    build_features(output_dir=output_dir)
    nodes_path = output_dir / "nodes.parquet"
    events_path = output_dir / "events_35d.parquet"
    splits_path = output_dir / "splits.parquet"
    features_path = output_dir / "X_base.npy"
    input_paths = (nodes_path, events_path, splits_path, features_path)
    inputs = _signatures(input_paths)
    if not force and (cached := _cache_hit(output_dir, inputs)) is not None:
        return {**cached, "cache_hit": True}

    structural_path = output_dir / STRUCTURAL_ARTIFACT
    behavioral_path = output_dir / BEHAVIORAL_ARTIFACT
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    try:
        _build_structural(connection, nodes_path, events_path, structural_path)
        train_ids_by_seed = [
            _train_ids(connection, splits_path, seed) for seed in EXPERIMENT_SEEDS
        ]
        structural_audit = audit_structural(
            connection, structural_path, splits_path, EXPERIMENT_SEEDS
        )
    finally:
        connection.close()

    # One thread makes the approximate index construction reproducible.
    import faiss

    faiss.omp_set_num_threads(1)
    features = np.load(features_path, mmap_mode="r")
    neighbors = np.empty(
        (len(EXPERIMENT_SEEDS), DATASET_CONTRACT.enrollments, K_MAX),
        dtype=np.int32,
    )
    for seed_index, train_ids in enumerate(train_ids_by_seed):
        neighbors[seed_index] = build_seed_neighbors(
            features[seed_index], train_ids, k_max=K_MAX
        )
    write_neighbors_atomic(behavioral_path, neighbors)
    behavioral_audit_indexed = audit_behavioral(neighbors, train_ids_by_seed)
    behavioral_audit = {
        str(seed): behavioral_audit_indexed[str(index)]
        for index, seed in enumerate(EXPERIMENT_SEEDS)
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "hyperedge_version": HYPEREDGE_VERSION,
        "structural": structural_audit,
        "behavioral_train_by_seed": behavioral_audit,
        "rules": {
            "minimum_structural_cardinality": 2,
            "object_types": ["video", "assignment", "forum"],
            "behavioral_reference_split": "train only",
            "behavioral_metric": "cosine on X_base",
            "behavioral_similarity_feature_set": "behavior",
            "behavioral_similarity_dimension": len(base_feature_columns()),
            "k_candidates": list(K_CANDIDATES),
        },
    }
    write_json_atomic(output_dir / AUDIT_ARTIFACT, audit)

    if _signatures(input_paths) != inputs:
        raise RuntimeError("An upstream artifact changed during Phase 4")
    artifact_paths = (
        structural_path,
        behavioral_path,
        output_dir / AUDIT_ARTIFACT,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "hyperedge_version": HYPEREDGE_VERSION,
        "dataset": DATASET_CONTRACT.dataset,
        "generated_utc": utc_now(),
        "cache_hit": False,
        "seeds": list(EXPERIMENT_SEEDS),
        "families": ["course", "object", "behavioral"],
        "k_candidates": list(K_CANDIDATES),
        "behavioral_array_layout": (
            "neighbors[seed_index, anchor_node_id, neighbor_rank]"
        ),
        "behavioral_array_shape": list(neighbors.shape),
        "behavioral_similarity_feature_set": "behavior",
        "behavioral_similarity_dimension": len(base_feature_columns()),
        "behavioral_array_dtype": str(neighbors.dtype),
        "hnsw": {
            "metric": "cosine (L2-normalized inner product)",
            "M": HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION,
            "ef_search": HNSW_EF_SEARCH,
            "threads": 1,
        },
        "inputs": inputs,
        "artifacts": _signatures(artifact_paths),
    }
    write_json_atomic(output_dir / MANIFEST_ARTIFACT, manifest)
    return manifest


# Mục đích: Tạo tên file H0 train nhất quán cho một seed.
# Đầu vào: Experiment seed.
# Đầu ra: Tên file dạng H0_train_seed_<seed>.npz.
def train_matrix_name(seed: int) -> str:
    return f"H0_train_seed_{seed}.npz"


# Mục đích: Liệt kê tất cả artifact phải tồn tại để graph cache hợp lệ.
# Đầu vào: Không có.
# Đầu ra: Tuple tên năm H0 matrix và các bảng metadata/evaluation/audit.
def _graph_artifact_names() -> tuple[str, ...]:
    matrices = tuple(train_matrix_name(seed) for seed in EXPERIMENT_SEEDS)
    return matrices + (
        TRAIN_NODE_INDEX_ARTIFACT,
        HYPEREDGE_METADATA_ARTIFACT,
        VALIDATION_MEMBERSHIPS_ARTIFACT,
        TEST_MEMBERSHIPS_ARTIFACT,
        GRAPH_AUDIT_ARTIFACT,
    )


# Mục đích: Kiểm tra graph cache khớp input và behavioral_k được yêu cầu.
# Đầu vào: Output directory, input signatures và behavioral_k.
# Đầu ra: Graph manifest hợp lệ hoặc None.
# Lưu ý: Mọi artifact đều được đối chiếu SHA-256 trước khi cache hit.
def _graph_cache_hit(
    output_dir: Path,
    inputs: dict[str, dict[str, Any]],
    behavioral_k: int,
) -> dict[str, Any] | None:
    path = output_dir / GRAPH_MANIFEST_ARTIFACT
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("graph_version") != GRAPH_VERSION
        or manifest.get("seeds") != list(EXPERIMENT_SEEDS)
        or manifest.get("behavioral_k") != behavioral_k
        or manifest.get("inputs") != inputs
    ):
        return None
    artifacts = manifest.get("artifacts", {})
    for name in _graph_artifact_names():
        path = output_dir / name
        if not path.is_file() or artifacts.get(name) != source_signature(
            path, include_hash=True
        ):
            return None
    return manifest


# Mục đích: Ghi SciPy CSR matrix thành NPZ nén theo cách atomic.
# Đầu vào: File đích và sparse incidence matrix.
# Đầu ra: Không trả dữ liệu; tạo file .npz hoàn chỉnh.
def _write_sparse_atomic(path: Path, matrix: sparse.csr_matrix) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as destination:
        sparse.save_npz(destination, matrix, compressed=True)
    temporary.replace(path)


# Mục đích: Tạo temporary DuckDB table chứa metadata theo đúng H0 column order.
# Đầu vào: Kết nối DuckDB đang dùng để materialize graph.
# Đầu ra: Không trả dữ liệu; tạo bảng tạm hyperedge_metadata.
def _create_metadata_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE hyperedge_metadata (
            seed INTEGER,
            hyperedge_id BIGINT,
            family VARCHAR,
            source_key VARCHAR,
            object_type VARCHAR,
            course_id VARCHAR,
            object_id VARCHAR,
            anchor_node_id BIGINT,
            size INTEGER,
            weight FLOAT
        )
        """
    )


# Mục đích: Chèn metadata hyperedge vào bảng tạm theo các batch nhỏ.
# Đầu vào: Kết nối và list tuple đúng schema metadata.
# Đầu ra: Không trả dữ liệu; cập nhật bảng tạm.
# Lưu ý: Batch 50.000 dòng tránh truyền một parameter array quá lớn.
def _insert_metadata(
    connection: duckdb.DuckDBPyConnection,
    rows: list[tuple[Any, ...]],
) -> None:
    batch_size = 50_000
    casts = (
        "INTEGER[]",
        "BIGINT[]",
        "VARCHAR[]",
        "VARCHAR[]",
        "VARCHAR[]",
        "VARCHAR[]",
        "VARCHAR[]",
        "BIGINT[]",
        "INTEGER[]",
        "FLOAT[]",
    )
    expressions = ", ".join(f"unnest(?::{cast})" for cast in casts)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        columns = [list(values) for values in zip(*batch, strict=True)]
        connection.execute(
            f"INSERT INTO hyperedge_metadata SELECT {expressions}", columns
        )


# Mục đích: Gom Course/Object memberships của train thành từng hyperedge.
# Đầu vào: Kết nối, memberships path, splits path và seed.
# Đầu ra: List tuple family/key/type/node_ids đã sort.
# Lưu ý: Loại hyperedge có ít hơn hai train nodes.
def _structural_groups(
    connection: duckdb.DuckDBPyConnection,
    memberships_path: Path,
    splits_path: Path,
    seed: int,
) -> list[tuple[Any, ...]]:
    return connection.execute(
        f"""
        SELECT
            m.family,
            m.course_id,
            m.object_id,
            m.object_type,
            list(m.node_id ORDER BY m.node_id) AS node_ids
        FROM read_parquet('{sql_path(memberships_path)}') m
        JOIN read_parquet('{sql_path(splits_path)}') s USING (node_id)
        WHERE s.seed=? AND s.experiment_split='train'
        GROUP BY m.family, m.course_id, m.object_id, m.object_type
        HAVING count(*) >= 2
        ORDER BY
            CASE m.family WHEN 'course' THEN 0 ELSE 1 END,
            m.course_id, m.object_type, m.object_id
        """,
        [seed],
    ).fetchall()


# Mục đích: Tạo khóa nguồn duy nhất cho Course hoặc Object hyperedge.
# Đầu vào: Family, course_id và object_id tùy chọn.
# Đầu ra: Course ID hoặc JSON composite key [course_id, object_id].
# Lưu ý: Composite key ngăn object trùng tên ở hai course bị nối nhầm.
def _source_key(family: str, course_id: str, object_id: str | None) -> str:
    if family == "course":
        return course_id
    return json.dumps([course_id, object_id], ensure_ascii=False, separators=(",", ":"))


# Mục đích: Tạo sparse train H0 và metadata cho một seed.
# Đầu vào: Kết nối, paths, neighbor tensor, seed index/seed và behavioral_k.
# Đầu ra: Audit shape, nnz, density và family counts của H0.
# Lưu ý: Row dùng local train index; metadata giữ mapping về global node IDs.
def _materialize_seed(
    connection: duckdb.DuckDBPyConnection,
    output_dir: Path,
    memberships_path: Path,
    splits_path: Path,
    neighbors: np.ndarray,
    seed_index: int,
    seed: int,
    behavioral_k: int,
) -> dict[str, Any]:
    train_ids = _train_ids(connection, splits_path, seed)
    global_to_local = np.full(DATASET_CONTRACT.enrollments, -1, dtype=np.int32)
    global_to_local[train_ids] = np.arange(train_ids.size, dtype=np.int32)

    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    metadata: list[tuple[Any, ...]] = []
    family_counts = {"course": 0, "object": 0, "behavioral": 0}
    edge_id = 0
    for family, course_id, object_id, object_type, node_ids in _structural_groups(
        connection, memberships_path, splits_path, seed
    ):
        global_ids = np.asarray(node_ids, dtype=np.int64)
        local_ids = global_to_local[global_ids]
        if np.any(local_ids < 0):
            raise RuntimeError("A structural train edge contains a non-train node")
        row_parts.append(local_ids)
        column_parts.append(np.full(local_ids.size, edge_id, dtype=np.int64))
        metadata.append(
            (
                seed,
                edge_id,
                family,
                _source_key(family, course_id, object_id),
                object_type,
                course_id,
                object_id,
                None,
                int(local_ids.size),
                1.0,
            )
        )
        family_counts[family] += 1
        edge_id += 1

    behavioral_edges = np.concatenate(
        (
            train_ids[:, None],
            neighbors[seed_index, train_ids, :behavioral_k],
        ),
        axis=1,
    )
    behavioral_edges.sort(axis=1)
    unique_edges, representative_indices = np.unique(
        behavioral_edges, axis=0, return_index=True
    )
    behavioral_local = global_to_local[unique_edges]
    if np.any(behavioral_local < 0):
        raise RuntimeError("A behavioral train edge contains a non-train node")
    behavioral_ids = np.arange(
        edge_id, edge_id + unique_edges.shape[0], dtype=np.int64
    )
    row_parts.append(behavioral_local.reshape(-1))
    column_parts.append(
        np.repeat(behavioral_ids, behavioral_k + 1)
    )
    representative_anchors = train_ids[representative_indices]
    metadata.extend(
        (
            seed,
            int(current_id),
            "behavioral",
            f"anchor:{int(anchor)}",
            None,
            None,
            None,
            int(anchor),
            behavioral_k + 1,
            1.0,
        )
        for current_id, anchor in zip(
            behavioral_ids, representative_anchors, strict=True
        )
    )
    family_counts["behavioral"] = int(unique_edges.shape[0])
    edge_id += int(unique_edges.shape[0])

    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    matrix = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, columns)),
        shape=(train_ids.size, edge_id),
        dtype=np.uint8,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    if matrix.nnz != rows.size or matrix.data.min(initial=1) != 1:
        raise RuntimeError("H0 must contain unique binary incidences")
    column_sizes = np.asarray(matrix.sum(axis=0)).reshape(-1)
    if column_sizes.size != edge_id or np.any(column_sizes < 2):
        raise RuntimeError("H0 contains an empty or singleton hyperedge")
    _write_sparse_atomic(output_dir / train_matrix_name(seed), matrix)
    _insert_metadata(connection, metadata)
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "families": family_counts,
        "minimum_edge_size": int(column_sizes.min()),
        "maximum_edge_size": int(column_sizes.max()),
        "train_nodes": int(train_ids.size),
    }


# Mục đích: Ghi mapping local H0 row sang global node_id cho mọi seed.
# Đầu vào: Kết nối, split path và file Parquet đích.
# Đầu ra: Không trả dữ liệu; tạo train_node_index.parquet.
def _write_train_node_index(
    connection: duckdb.DuckDBPyConnection,
    splits_path: Path,
    destination: Path,
) -> None:
    seeds = ", ".join(str(seed) for seed in EXPERIMENT_SEEDS)
    query = f"""
        SELECT
            seed,
            (row_number() OVER (PARTITION BY seed ORDER BY node_id) - 1)::BIGINT
                AS local_node_id,
            node_id
        FROM read_parquet('{sql_path(splits_path)}')
        WHERE experiment_split='train' AND seed IN ({seeds})
        ORDER BY seed, local_node_id
    """
    copy_parquet_atomic(connection, query, destination)


# Mục đích: Ghi bảng metadata theo đúng seed và hyperedge_id column order.
# Đầu vào: Kết nối có bảng tạm metadata và file đích.
# Đầu ra: Không trả dữ liệu; tạo hyperedges.parquet.
def _write_hyperedge_metadata(
    connection: duckdb.DuckDBPyConnection, destination: Path
) -> None:
    copy_parquet_atomic(
        connection,
        "SELECT * FROM hyperedge_metadata ORDER BY seed, hyperedge_id",
        destination,
    )


# Mục đích: Sinh SQL mô tả local graph cho từng validation/test target.
# Đầu vào: Membership path, split path, tên split và behavioral_k.
# Đầu ra: Chuỗi SQL tạo Course/Object/Behavioral local memberships.
# Lưu ý: Reference node luôn thuộc train; không tạo cạnh giữa hai target.
def _local_membership_query(
    memberships_path: Path,
    splits_path: Path,
    experiment_split: str,
    behavioral_k: int,
) -> str:
    if experiment_split not in {"validation", "test"}:
        raise ValueError("Local memberships are only defined for validation/test")
    memberships = sql_path(memberships_path)
    splits = sql_path(splits_path)
    return f"""
        WITH train_groups AS (
            SELECT
                s.seed,
                m.family,
                m.course_id,
                m.object_id,
                m.object_type,
                count(*)::INTEGER AS reference_size,
                min(m.node_id)::BIGINT AS singleton_reference_node_id
            FROM read_parquet('{memberships}') m
            JOIN read_parquet('{splits}') s USING (node_id)
            WHERE s.experiment_split='train'
            GROUP BY s.seed, m.family, m.course_id, m.object_id, m.object_type
        ),
        target_structural AS (
            SELECT
                s.seed,
                s.node_id AS target_node_id,
                m.family,
                CASE
                    WHEN m.family='course' THEN m.course_id::VARCHAR
                    ELSE to_json([m.course_id, m.object_id])::VARCHAR
                END::VARCHAR AS source_key,
                m.object_type,
                h.hyperedge_id AS train_hyperedge_id,
                CASE WHEN g.reference_size=1
                    THEN g.singleton_reference_node_id END
                    AS singleton_reference_node_id,
                NULL::BIGINT AS behavioral_anchor_node_id,
                (g.reference_size + 1)::INTEGER AS size,
                1.0::FLOAT AS weight
            FROM read_parquet('{memberships}') m
            JOIN read_parquet('{splits}') s USING (node_id)
            JOIN train_groups g
              ON g.seed=s.seed AND g.family=m.family
             AND g.course_id=m.course_id
             AND g.object_id IS NOT DISTINCT FROM m.object_id
             AND g.object_type IS NOT DISTINCT FROM m.object_type
            LEFT JOIN hyperedge_metadata h
              ON h.seed=s.seed AND h.family=m.family
             AND h.course_id=m.course_id
             AND h.object_id IS NOT DISTINCT FROM m.object_id
             AND h.object_type IS NOT DISTINCT FROM m.object_type
            WHERE s.experiment_split='{experiment_split}'
              AND m.family IN ('course', 'object')
        ),
        target_behavioral AS (
            SELECT
                seed,
                node_id AS target_node_id,
                'behavioral'::VARCHAR AS family,
                'anchor:' || node_id::VARCHAR AS source_key,
                NULL::VARCHAR AS object_type,
                NULL::BIGINT AS train_hyperedge_id,
                NULL::BIGINT AS singleton_reference_node_id,
                node_id::BIGINT AS behavioral_anchor_node_id,
                {behavioral_k + 1}::INTEGER AS size,
                1.0::FLOAT AS weight
            FROM read_parquet('{splits}')
            WHERE experiment_split='{experiment_split}'
        ),
        combined AS (
            SELECT * FROM target_structural
            UNION ALL
            SELECT * FROM target_behavioral
        )
        SELECT
            seed,
            '{experiment_split}'::VARCHAR AS experiment_split,
            target_node_id,
            (row_number() OVER (
                PARTITION BY seed, target_node_id
                ORDER BY CASE family
                    WHEN 'course' THEN 0 WHEN 'object' THEN 1 ELSE 2 END,
                    source_key
            ) - 1)::BIGINT AS local_hyperedge_id,
            family,
            source_key,
            object_type,
            train_hyperedge_id,
            singleton_reference_node_id,
            behavioral_anchor_node_id,
            size,
            weight
        FROM combined
        ORDER BY seed, target_node_id, local_hyperedge_id
    """


# Mục đích: Kiểm tra local memberships đúng split và chỉ trỏ tới train reference.
# Đầu vào: Kết nối, local membership path và split path.
# Đầu ra: Dictionary audit theo seed.
# Lưu ý: Phát hiện edge size <2, sai target split hoặc reference ngoài train.
def _audit_local_memberships(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    splits_path: Path,
) -> dict[str, Any]:
    memberships = sql_path(path)
    splits = sql_path(splits_path)
    rows = connection.execute(
        f"""
        SELECT
            seed,
            experiment_split,
            count(*) AS local_hyperedges,
            count(DISTINCT target_node_id) AS targets,
            count(*) FILTER (WHERE family='course') AS course,
            count(*) FILTER (WHERE family='object') AS object,
            count(*) FILTER (WHERE family='behavioral') AS behavioral,
            count(*) FILTER (
                WHERE singleton_reference_node_id IS NOT NULL
            ) AS singleton_reference_edges,
            min(size) AS minimum_size,
            max(size) AS maximum_size
        FROM read_parquet('{memberships}')
        GROUP BY seed, experiment_split
        ORDER BY seed
        """
    ).fetchall()
    invalid = connection.execute(
        f"""
        SELECT count(*)
        FROM read_parquet('{memberships}') m
        LEFT JOIN read_parquet('{splits}') target
          ON target.seed=m.seed AND target.node_id=m.target_node_id
        LEFT JOIN read_parquet('{splits}') reference
          ON reference.seed=m.seed
         AND reference.node_id=m.singleton_reference_node_id
        LEFT JOIN hyperedge_metadata h
          ON h.seed=m.seed AND h.hyperedge_id=m.train_hyperedge_id
        WHERE target.experiment_split IS DISTINCT FROM m.experiment_split
           OR m.size < 2
           OR (m.family='behavioral'
               AND m.behavioral_anchor_node_id != m.target_node_id)
           OR (m.singleton_reference_node_id IS NOT NULL
               AND reference.experiment_split IS DISTINCT FROM 'train')
           OR (m.train_hyperedge_id IS NOT NULL
               AND (h.hyperedge_id IS NULL
                    OR h.family IS DISTINCT FROM m.family
                    OR h.family='behavioral'))
           OR (m.family IN ('course', 'object')
               AND ((m.train_hyperedge_id IS NULL)
                    = (m.singleton_reference_node_id IS NULL)))
        """
    ).fetchone()[0]
    if invalid:
        raise RuntimeError(f"Found {invalid} invalid local hyperedges")
    keys = (
        "seed",
        "experiment_split",
        "local_hyperedges",
        "targets",
        "course",
        "object",
        "behavioral",
        "singleton_reference_edges",
        "minimum_size",
        "maximum_size",
    )
    return {
        str(row[0]): dict(zip(keys[1:], row[1:], strict=True)) for row in rows
    }


# Mục đích: Materialize H0 train và compact local memberships cho validation/test.
# Đầu vào: Output directory, behavioral_k và cờ force rebuild.
# Đầu ra: Graph manifest chứa artifact hashes và audit toàn bộ năm seed.
# Lưu ý: H0 là binary CSR; evaluation graph được dựng lại khi cần để tiết kiệm đĩa.
def build_initial_hypergraph(
    output_dir: Path = PROCESSED_DATA_DIR,
    *,
    behavioral_k: int = DEFAULT_K,
    force: bool = False,
) -> dict[str, Any]:
    if behavioral_k not in K_CANDIDATES:
        raise ValueError(f"behavioral_k must be one of {K_CANDIDATES}")
    output_dir = require_project_path(output_dir)
    build_hyperedges(output_dir=output_dir)
    memberships_path = output_dir / STRUCTURAL_ARTIFACT
    neighbors_path = output_dir / BEHAVIORAL_ARTIFACT
    splits_path = output_dir / "splits.parquet"
    features_path = output_dir / "X_base.npy"
    input_paths = (memberships_path, neighbors_path, splits_path, features_path)
    inputs = _signatures(input_paths)
    if not force and (
        cached := _graph_cache_hit(output_dir, inputs, behavioral_k)
    ) is not None:
        return {**cached, "cache_hit": True}

    with np.load(neighbors_path, allow_pickle=False) as archive:
        neighbors = archive["neighbors"]
    features = np.load(features_path, mmap_mode="r")
    expected_feature_shape = (
        len(EXPERIMENT_SEEDS),
        DATASET_CONTRACT.enrollments,
        len(base_feature_columns()),
    )
    if features.shape != expected_feature_shape:
        raise RuntimeError(f"Unexpected X_base shape: {features.shape}")

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    _create_metadata_table(connection)
    train_audit: dict[str, Any] = {}
    try:
        for seed_index, seed in enumerate(EXPERIMENT_SEEDS):
            train_audit[str(seed)] = _materialize_seed(
                connection,
                output_dir,
                memberships_path,
                splits_path,
                neighbors,
                seed_index,
                seed,
                behavioral_k,
            )
        _write_train_node_index(
            connection, splits_path, output_dir / TRAIN_NODE_INDEX_ARTIFACT
        )
        _write_hyperedge_metadata(
            connection, output_dir / HYPEREDGE_METADATA_ARTIFACT
        )
        for split, name in (
            ("validation", VALIDATION_MEMBERSHIPS_ARTIFACT),
            ("test", TEST_MEMBERSHIPS_ARTIFACT),
        ):
            copy_parquet_atomic(
                connection,
                _local_membership_query(
                    memberships_path, splits_path, split, behavioral_k
                ),
                output_dir / name,
            )
        local_audit = {
            "validation": _audit_local_memberships(
                connection,
                output_dir / VALIDATION_MEMBERSHIPS_ARTIFACT,
                splits_path,
            ),
            "test": _audit_local_memberships(
                connection,
                output_dir / TEST_MEMBERSHIPS_ARTIFACT,
                splits_path,
            ),
        }
    finally:
        connection.close()

    audit = {
        "schema_version": SCHEMA_VERSION,
        "graph_version": GRAPH_VERSION,
        "behavioral_k": behavioral_k,
        "train_by_seed": train_audit,
        "local_evaluation": local_audit,
        "rules": {
            "incidence": "binary CSR",
            "hyperedge_weight": 1.0,
            "minimum_hyperedge_size": 2,
            "evaluation_mode": "one target with train references only",
            "feature_layout": "X_base[seed_index, global_node_id, feature]",
            "behavioral_similarity_feature_set": "behavior",
        },
    }
    write_json_atomic(output_dir / GRAPH_AUDIT_ARTIFACT, audit)
    if _signatures(input_paths) != inputs:
        raise RuntimeError("A Phase 4 artifact changed during Phase 5")

    artifact_paths = tuple(output_dir / name for name in _graph_artifact_names())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "graph_version": GRAPH_VERSION,
        "dataset": DATASET_CONTRACT.dataset,
        "generated_utc": utc_now(),
        "cache_hit": False,
        "seeds": list(EXPERIMENT_SEEDS),
        "behavioral_k": behavioral_k,
        "behavioral_k_status": "default candidate; final choice requires validation",
        "behavioral_similarity_feature_set": "behavior",
        "behavioral_similarity_dimension": len(base_feature_columns()),
        "families": ["course", "object", "behavioral"],
        "matrix_format": "SciPy CSR uint8 saved with save_npz",
        "local_membership_encoding": {
            "structural": (
                "train_hyperedge_id, or singleton_reference_node_id when the "
                "object has exactly one train reference"
            ),
            "behavioral": (
                "behavioral_anchor_node_id indexes behavioral_neighbors.npz"
            ),
        },
        "inputs": inputs,
        "artifacts": _signatures(artifact_paths),
    }
    write_json_atomic(output_dir / GRAPH_MANIFEST_ARTIFACT, manifest)
    return manifest
