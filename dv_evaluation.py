"""Rand-Index (from scratch), eine kompakte Single-Linkage-Referenzimplementierung NUR
für den Methodenvergleich dieser Demo (nicht die volle Lance-Williams-Allgemeinheit aus
agglomerative-demo/ag_algorithm.py - bewusst kein Cross-Import, siehe Modul-Docstring
dort: jede Demo bleibt eigenstaendig lauffaehig), sowie der Bisecting-k-Means-vs.-
Single-Linkage-Vergleich - das "was kann Bisecting k-Means, was Single-Linkage nicht
kann"-Aequivalent zu jeder vorherigen Demo dieser Reihe."""

import numpy as np

from dv_algorithm import labels_at_step, run


def rand_index(true_labels, pred_labels):
    """Anteil der Punktpaare, bei denen beide Partitionen uebereinstimmen (entweder
    beide im selben Cluster oder beide in unterschiedlichen)."""
    true_arr = np.asarray(true_labels)
    pred_arr = np.asarray(pred_labels)
    n = len(true_arr)
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    same_true = (true_arr[:, None] == true_arr[None, :])[iu]
    same_pred = (pred_arr[:, None] == pred_arr[None, :])[iu]
    return float(np.mean(same_true == same_pred))


def single_linkage_labels(data, k):
    """Naive O(n^3) Single-Linkage-Referenz (staerkstes Kriterium fuer nicht-konvexe/
    zusammenhaengende Formen, siehe agglomerative-demo) - NUR fuer den Methodenvergleich
    dieser Demo. Fusioniert wiederholt die zwei Cluster mit dem kleinsten minimalen
    Punktabstand, bis genau k Cluster aktiv sind."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    diffs = data[:, None, :] - data[None, :, :]
    dist = np.sqrt((diffs ** 2).sum(axis=2))

    clusters = {i: {i} for i in range(n)}
    while len(clusters) > k:
        ids = list(clusters.keys())
        best_pair, best_d = None, np.inf
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                d = min(dist[p, q] for p in clusters[a] for q in clusters[b])
                if d < best_d:
                    best_d, best_pair = d, (a, b)
        a, b = best_pair
        clusters[a] |= clusters[b]
        del clusters[b]

    labels = np.empty(n, dtype=int)
    for new_label, members in enumerate(clusters.values()):
        for p in members:
            labels[p] = new_label
    return labels


def method_comparison(data, true_labels, target_k, split_criterion, seed):
    """Rand-Index von Bisecting k-Means (beim eingestellten `target_k` und
    `split_criterion`) gegen die Single-Linkage-Referenz, beide beim GLEICHEN Ziel-k -
    die front-and-center 📐-Kernaussage dieser Demo."""
    n = len(data)
    result = run(data, target_k, split_criterion, seed)
    bisecting_labels = labels_at_step(n, result.splits, target_k - 2)
    baseline_labels = single_linkage_labels(data, target_k)
    return {
        "bisecting": rand_index(true_labels, bisecting_labels),
        "single_linkage_baseline": rand_index(true_labels, baseline_labels),
    }
