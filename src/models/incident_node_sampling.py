"""Positive and negative node sampling for selected hyperedges."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SampledMemberships:
    edge_ids: np.ndarray
    positive_nodes: tuple[np.ndarray, ...]
    negative_nodes: tuple[np.ndarray, ...]


def _negative_nodes(
    node_count: int,
    incident_nodes: np.ndarray,
    count: int,
    generator: np.random.Generator,
) -> np.ndarray:
    available = node_count - incident_nodes.size
    target = min(count, available)
    if target == 0:
        return np.empty(0, dtype=np.int64)
    incident = set(int(node) for node in incident_nodes)
    selected: set[int] = set()
    while len(selected) < target:
        draws = generator.integers(
            0, node_count, size=max(2 * (target - len(selected)), 8)
        )
        selected.update(int(node) for node in draws if int(node) not in incident)
    values = np.fromiter(selected, dtype=np.int64)
    if values.size > target:
        values = generator.choice(values, size=target, replace=False)
    return np.sort(values)


def sample_incident_nodes(
    incidence: sparse.spmatrix,
    edge_ids: np.ndarray,
    *,
    positive_count: int,
    negative_count: int,
    generator: np.random.Generator,
) -> SampledMemberships:
    """Sample incident positives and guaranteed non-incident negatives per edge."""

    if positive_count <= 0 or negative_count < 0:
        raise ValueError("positive_count must be positive and negative_count non-negative")
    matrix = incidence.tocsc()
    edge_ids = np.asarray(edge_ids, dtype=np.int64).reshape(-1)
    if edge_ids.size == 0 or np.any(edge_ids < 0) or np.any(edge_ids >= matrix.shape[1]):
        raise ValueError("edge_ids contain an invalid hyperedge")
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    for edge_id in edge_ids:
        start, stop = matrix.indptr[edge_id : edge_id + 2]
        incident = matrix.indices[start:stop].astype(np.int64, copy=False)
        if incident.size == 0:
            raise ValueError("Cannot sample an empty hyperedge")
        count = min(positive_count, incident.size)
        positive = np.sort(generator.choice(incident, size=count, replace=False))
        negative = _negative_nodes(
            matrix.shape[0], incident, negative_count, generator
        )
        if np.intersect1d(positive, negative).size:
            raise RuntimeError("Positive and negative samples overlap")
        positives.append(positive)
        negatives.append(negative)
    return SampledMemberships(edge_ids, tuple(positives), tuple(negatives))
