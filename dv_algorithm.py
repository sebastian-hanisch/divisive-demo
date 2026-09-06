"""Bisecting k-Means (Divisive Clustering) from scratch, mit vollständigem
Split-Protokoll, damit die App Split für Split (nicht nur das Endergebnis) durchblättern
kann - analog zum Merge-Protokoll in agglomerative-demo/ag_algorithm.py, nur umgekehrt
wachsend: statt n Singleton-Clustern zu verschmelzen, wird EIN Cluster (alle Punkte)
wiederholt in zwei geteilt.

Jeder Split ist ein frisches, kleines 2-Means (mehrere interne Neustarts) - bewusst ohne
Cross-Import aus kmeans-demo (Konvention dieser Reihe: jede Demo eigenstaendig lauffaehig).
sklearn.cluster.BisectingKMeans dient in tests/ nur als unabhaengiger Kreuzvergleich."""

from dataclasses import dataclass

import numpy as np

from dv_constants import N_RESTARTS


@dataclass(frozen=True)
class Split:
    step: int  # Reihenfolge des Splits (0-indiziert)
    parent_cluster: int  # ID des gesplitteten Clusters
    child_a: int  # neue Cluster-ID (erstes Kind)
    child_b: int  # neue Cluster-ID (zweites Kind)
    members_a: tuple  # Punktindizes, die zu child_a gehen
    members_b: tuple  # Punktindizes, die zu child_b gehen
    sse_before: float  # Fehlerquadratsumme des Elternclusters vor dem Split
    sse_after: float  # Summe der Fehlerquadratsummen beider Kinder (immer <= sse_before)


@dataclass(frozen=True)
class RunResult:
    splits: tuple  # max_k - 1 Split-Eintraege in chronologischer Reihenfolge
    n_points: int
    split_criterion: str

    @property
    def n_splits(self):
        return len(self.splits)


def _sse(data, indices, center=None):
    pts = data[list(indices)]
    if center is None:
        center = pts.mean(axis=0)
    return float(((pts - center) ** 2).sum())


def _kmeans2(data, indices, seed, n_restarts=N_RESTARTS, max_iter=100):
    """Frisches 2-Means (Lloyd's) auf einer Teilmenge der Punkte, mit `n_restarts`
    unabhaengigen Zufallsinitialisierungen (jeweils zwei verschiedene Datenpunkte als
    Startzentren) - behaelt den Lauf mit der niedrigsten End-SSE. Reduziert, beseitigt
    aber nicht das Lokale-Optima-Risiko aus kmeans-demo: bei sehr wenigen Neustarts
    kann ein Split trotzdem in einem schlechten lokalen Optimum landen."""
    rng = np.random.default_rng(seed)
    pts = data[list(indices)]
    n = len(pts)

    best_labels, best_sse = None, np.inf
    for _ in range(n_restarts):
        idx = rng.choice(n, size=2, replace=False)
        centers = pts[idx].copy()
        labels = None
        for _iteration in range(max_iter):
            d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = d2.argmin(axis=1)
            converged = labels is not None and np.array_equal(new_labels, labels)
            labels = new_labels
            if converged:
                break
            for c in range(2):
                mask = labels == c
                if mask.any():
                    centers[c] = pts[mask].mean(axis=0)
        if (labels == 0).all() or (labels == 1).all():
            continue  # entarteter Split (ein Kind leer) - verwerfen, anderen Neustart versuchen
        sse = sum(_sse(pts, np.where(labels == c)[0]) for c in range(2))
        if sse < best_sse:
            best_sse, best_labels = sse, labels

    if best_labels is None:
        # Alle Neustarts entartet (z.B. n=2 identische Punkte) - deterministischer Fallback:
        # erste Haelfte/zweite Haelfte der uebergebenen Reihenfolge.
        best_labels = np.array([0] * (n // 2) + [1] * (n - n // 2))

    local_a = [int(indices[i]) for i in range(n) if best_labels[i] == 0]
    local_b = [int(indices[i]) for i in range(n) if best_labels[i] == 1]
    return tuple(local_a), tuple(local_b)


def _select_parent(clusters, data, criterion):
    """Waehlt den zu splittenden Cluster: "size" (meiste Punkte) oder "sse" (hoechste
    Fehlerquadratsumme zum eigenen Schwerpunkt) - Analogon zu agglomerative-demos
    `linkage`, hier aber die Split-Auswahl statt das Fusionskriterium."""
    candidates = [cid for cid, members in clusters.items() if len(members) >= 2]
    if criterion == "size":
        return max(candidates, key=lambda cid: len(clusters[cid]))
    return max(candidates, key=lambda cid: _sse(data, clusters[cid]))


def run(data, max_k, split_criterion, seed):
    """Fuehrt Bisecting k-Means vollstaendig protokolliert aus, bis `max_k` Cluster aktiv
    sind. Ein kleineres Ziel-k ist einfach ein Praefix der zurueckgegebenen Splits (siehe
    `labels_at_step`) - exakt das Muster aus agglomerative-demos
    `step_for_target_k`/`labels_at_step`, damit das Dendrogramm-Layout beim
    Schrittregler nicht springt."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    clusters = {0: tuple(range(n))}
    next_id = 1
    splits = []

    for step in range(max_k - 1):
        parent = _select_parent(clusters, data, split_criterion)
        parent_members = clusters[parent]
        sse_before = _sse(data, parent_members)

        split_seed = (int(seed) * 1_000_003 + step) % (2**32 - 1)
        members_a, members_b = _kmeans2(data, parent_members, split_seed)

        child_a, child_b = next_id, next_id + 1
        next_id += 2
        sse_after = _sse(data, members_a) + _sse(data, members_b)

        splits.append(
            Split(step, parent, child_a, child_b, members_a, members_b, sse_before, sse_after)
        )

        del clusters[parent]
        clusters[child_a] = members_a
        clusters[child_b] = members_b

    return RunResult(splits=tuple(splits), n_points=n, split_criterion=split_criterion)


def labels_at_step(n_points, splits, step):
    """Partition nach genau `step + 1` angewendeten Splits (0-indiziert), als fortlaufend
    nummerierte Labels 0..k-1 (Reihenfolge nach erstem Auftreten, nur fuer stabile
    Einfaerbung - keine inhaltliche Bedeutung). `step = -1` liefert die Wurzel (ein
    einziges Cluster)."""
    label = [0] * n_points
    for split in splits[: step + 1]:
        for p in split.members_a:
            label[p] = split.child_a
        for p in split.members_b:
            label[p] = split.child_b

    seen = {}
    result = []
    for l in label:
        if l not in seen:
            seen[l] = len(seen)
        result.append(seen[l])
    return tuple(result)


def total_sse_at_step(data, n_points, splits, step):
    """Gesamt-Fehlerquadratsumme ueber alle zu diesem Schritt aktiven Cluster - nach
    jedem weiteren Split beweisbar monoton nicht steigend (ein 2-Means-Split kann die
    SSE eines Clusters nur senken oder gleich lassen), analog zu kmeans-demos
    Inertia-Monotonie ueber Iterationen, hier aber ueber Splits."""
    data = np.asarray(data, dtype=float)
    labels = labels_at_step(n_points, splits, step)
    labels_arr = np.array(labels)
    total = 0.0
    for cid in set(labels):
        idx = np.where(labels_arr == cid)[0]
        total += _sse(data, idx)
    return total
