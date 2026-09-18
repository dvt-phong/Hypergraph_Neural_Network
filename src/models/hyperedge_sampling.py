"""Balanced hyperedge sampling by semantic family and cardinality bucket."""
from __future__ import annotations

from collections import Counter

import numpy as np


SIZE_BUCKETS = ("small", "medium", "large")


def cardinality_buckets(sizes: np.ndarray) -> np.ndarray:
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    if sizes.size == 0 or np.any(sizes < 2):
        raise ValueError("sizes must contain hyperedge cardinalities >= 2")
    return np.where(sizes <= 10, "small", np.where(sizes <= 100, "medium", "large"))


def sample_balanced_hyperedges(
    families: np.ndarray,
    sizes: np.ndarray,
    sample_size: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Round-robin sample across non-empty family/size strata."""

    families = np.asarray(families).astype(str).reshape(-1)
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    if families.shape != sizes.shape or families.size == 0:
        raise ValueError("families and sizes must be non-empty with equal shape")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    buckets = cardinality_buckets(sizes)
    strata: dict[tuple[str, str], np.ndarray] = {}
    for family, bucket in sorted(set(zip(families, buckets, strict=True))):
        indices = np.flatnonzero((families == family) & (buckets == bucket))
        strata[(family, bucket)] = generator.permutation(indices)

    target = min(sample_size, families.size)
    cursors = {key: 0 for key in strata}
    selected: list[int] = []
    while len(selected) < target:
        progressed = False
        for key in strata:
            cursor = cursors[key]
            if cursor < strata[key].size:
                selected.append(int(strata[key][cursor]))
                cursors[key] += 1
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break
    return np.asarray(selected, dtype=np.int64)


def sampled_strata_audit(
    edge_ids: np.ndarray,
    families: np.ndarray,
    sizes: np.ndarray,
) -> dict[str, int]:
    buckets = cardinality_buckets(np.asarray(sizes)[edge_ids])
    selected_families = np.asarray(families).astype(str)[edge_ids]
    counts = Counter(
        f"{family}:{bucket}"
        for family, bucket in zip(selected_families, buckets, strict=True)
    )
    return dict(sorted(counts.items()))
