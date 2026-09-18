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
    allowed_nodes: np.ndarray | None = None,
) -> np.ndarray:
    if allowed_nodes is not None:
        candidates = np.setdiff1d(
            np.asarray(allowed_nodes, dtype=np.int64),
            incident_nodes,
            assume_unique=False,
        )
        target = min(count, candidates.size)
        if target == 0:
            return np.empty(0, dtype=np.int64)
        return np.sort(generator.choice(candidates, size=target, replace=False))
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
    node_groups: np.ndarray | None = None,
    edge_groups: np.ndarray | None = None,
) -> SampledMemberships:
    """Sample incident positives and guaranteed non-incident negatives per edge."""

    if positive_count <= 0 or negative_count < 0:
        raise ValueError("positive_count must be positive and negative_count non-negative")
    matrix = incidence.tocsc()
    edge_ids = np.asarray(edge_ids, dtype=np.int64).reshape(-1)
    if edge_ids.size == 0 or np.any(edge_ids < 0) or np.any(edge_ids >= matrix.shape[1]):
        raise ValueError("edge_ids contain an invalid hyperedge")
    if (node_groups is None) != (edge_groups is None):
        raise ValueError("node_groups and edge_groups must be provided together")
    if node_groups is not None:
        node_groups = np.asarray(node_groups).reshape(-1)
        edge_groups = np.asarray(edge_groups).reshape(-1)
        if node_groups.size != matrix.shape[0] or edge_groups.size != matrix.shape[1]:
            raise ValueError("Group arrays must match incidence dimensions")
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    for edge_id in edge_ids:
        start, stop = matrix.indptr[edge_id : edge_id + 2]
        incident = matrix.indices[start:stop].astype(np.int64, copy=False)
        if incident.size == 0:
            raise ValueError("Cannot sample an empty hyperedge")
        count = min(positive_count, incident.size)
        positive = np.sort(generator.choice(incident, size=count, replace=False))
        allowed = None
        if node_groups is not None and edge_groups is not None:
            allowed = np.flatnonzero(node_groups == edge_groups[edge_id])
        negative = _negative_nodes(
            matrix.shape[0], incident, negative_count, generator, allowed
        )
        if np.intersect1d(positive, negative).size:
            raise RuntimeError("Positive and negative samples overlap")
        positives.append(positive)
        negatives.append(negative)
    return SampledMemberships(edge_ids, tuple(positives), tuple(negatives))
