"""Orakel-Tests für Bisecting k-Means (unabhängige Rechenwege):

1. Exakte Aufzählung aller 2^(n-1)-1 Zweiteilungen auf kleinen Instanzen: die Heuristik darf nie besser
   sein als das Optimum, und jeder Split muss ein Lloyd-Fixpunkt sein (2-Means ist nur eine Näherung).
2. Split-Protokoll nachgerechnet: Elternwahl nach Kriterium, SSE-Buchführung, Kinder = Eltern, Label-Anzahl.
3. Single-Linkage-Referenz gegen scipy.cluster.hierarchy (eindeutige Abstände: Partition identisch).
4. Rand-Index gegen sklearn.metrics.rand_score."""

import numpy as np
import pytest

from dv_algorithm import labels_at_step, run, total_sse_at_step
from dv_evaluation import rand_index, single_linkage_labels
from dv_scenario import generate_instance


def _sse(points):
    return float(((points - points.mean(0)) ** 2).sum()) if len(points) else 0.0


def _best_bisection_sse(points):
    n = len(points)
    best = np.inf
    for mask in range(1, 2 ** (n - 1)):
        a = [i for i in range(n - 1) if (mask >> i) & 1]
        b = [i for i in range(n) if i not in a]
        best = min(best, _sse(points[a]) + _sse(points[b]))
    return best


def test_first_split_is_valid_lloyd_fixpoint_and_not_better_than_exact_optimum():
    rng = np.random.default_rng(11)
    for trial in range(40):
        n = int(rng.integers(3, 11))
        points = rng.integers(0, 4, size=(n, 2)).astype(float) if trial % 4 == 0 else rng.normal(size=(n, 2))
        split = run(points, 2, "sse", trial).splits[0]
        a, b = list(split.members_a), list(split.members_b)
        assert sorted(a + b) == list(range(n)) and a and b
        assert split.sse_before == pytest.approx(_sse(points), abs=1e-9)
        assert split.sse_after == pytest.approx(_sse(points[a]) + _sse(points[b]), abs=1e-9)
        assert split.sse_after >= _best_bisection_sse(points) - 1e-9
        if len({tuple(p) for p in points}) == n:  # ohne Duplikate: Lloyd-Fixpunkt
            ca, cb = points[a].mean(0), points[b].mean(0)
            labels = np.array([0 if i in a else 1 for i in range(n)])
            nearest = np.stack([((points - ca) ** 2).sum(1), ((points - cb) ** 2).sum(1)], axis=1).argmin(1)
            assert np.array_equal(nearest, labels)


def test_split_sequence_reproduces_parent_choice_and_sse_bookkeeping():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(5)
    for trial in range(30):
        criterion = ("size", "sse")[trial % 2]
        instance = generate_instance(
            int(rng.integers(30, 80)), int(rng.integers(2, 6)), float(rng.uniform(0.05, 0.9)),
            float(rng.uniform(0, 1)), str(rng.choice(["blobs", "moons"])), int(rng.integers(0, 1000)),
        )
        data = instance.as_array()
        result = run(data, int(rng.integers(2, 9)), criterion, trial)
        active = {0: tuple(range(len(data)))}
        for sp in result.splits:
            candidates = {c: m for c, m in active.items() if len(m) >= 2}
            key = (lambda c: len(candidates[c])) if criterion == "size" else (lambda c: _sse(data[list(candidates[c])]))
            assert key(sp.parent_cluster) >= max(key(c) for c in candidates) - 1e-9
            parent = active.pop(sp.parent_cluster)
            assert set(sp.members_a) | set(sp.members_b) == set(parent)
            assert not set(sp.members_a) & set(sp.members_b)
            assert sp.sse_before == pytest.approx(_sse(data[list(parent)]), abs=1e-9)
            assert sp.sse_after == pytest.approx(_sse(data[list(sp.members_a)]) + _sse(data[list(sp.members_b)]), abs=1e-9)
            active[sp.child_a], active[sp.child_b] = sp.members_a, sp.members_b
        for step in range(-1, result.n_splits):
            labels = np.array(labels_at_step(len(data), result.splits, step))
            assert len(set(labels)) == step + 2
            expected = sum(_sse(data[labels == c]) for c in set(labels))
            assert total_sse_at_step(data, len(data), result.splits, step) == pytest.approx(expected, abs=1e-8)
        final = labels_at_step(len(data), result.splits, result.n_splits - 1)
        assert rand_index(instance.true_labels, final) == pytest.approx(
            metrics.rand_score(instance.true_labels, final), abs=1e-12
        )


def test_single_linkage_reference_matches_scipy():
    hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(2)
    for _ in range(40):
        n = int(rng.integers(3, 30))
        k = int(rng.integers(1, min(n, 8) + 1))
        data = rng.normal(size=(n, 2))
        reference = hierarchy.fcluster(hierarchy.linkage(data, "single"), k, "maxclust")
        assert metrics.adjusted_rand_score(reference, single_linkage_labels(data, k)) == 1.0
