"""Zufällige 2D-Punktwolken für die Divisive-Clustering-Demo: entweder k Gauß-Cluster
("blobs") oder k nicht-konvexe Halbkreis-Bögen ("moons", bei k=2 das klassische
Zwei-Halbmonde-Beispiel, bei k>2 Blütenblätter auf einem Ring wie in dbscan-demo/
spectral-demo). `spread_imbalance` macht Gruppe 0 größer UND enger (viele Punkte, wenig
Streuung -> niedrige Fehlerquadratsumme) und Gruppe 1 kleiner UND diffuser (wenige
Punkte, viel Streuung -> hohe Fehlerquadratsumme) - der Mechanismus, der "size"- und
"sse"-Split-Kriterium bei Bisecting k-Means auseinanderlaufen lässt. Wirkt unter BEIDEN
Formen, nur bei k>=2 (sonst ohne Effekt, da es dann nur eine Gruppe gibt)."""

from dataclasses import dataclass

import numpy as np

RING_RADIUS = 3.0
ARC_RADIUS = 2.5
ARC_RING_RADIUS = 6.5
MIN_STD_FRACTION = 0.05
MIN_SHARE_FRACTION = 0.12


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple  # ((x, y), ...)
    true_labels: tuple  # Gruppenindex
    shape: str  # "blobs" oder "moons"
    k: int

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _shares_and_scales(k, spread_imbalance):
    """Gruppe 0: waechst mit `spread_imbalance` (mehr Punkte), wird enger (kleinere
    Streuung). Gruppe 1 (falls k>=2): schrumpft (weniger Punkte), wird diffuser
    (groessere Streuung). Uebrige Gruppen bleiben bei der Baseline."""
    shares = np.ones(k)
    scales = np.ones(k)
    if k >= 2:
        shares[0] = 1 + spread_imbalance * 1.6
        shares[1] = max(1 - spread_imbalance * 0.75, MIN_SHARE_FRACTION)
        scales[0] = max(1 - spread_imbalance * 0.7, 0.3)
        scales[1] = 1 + spread_imbalance * 2.8
    shares = shares / shares.sum()
    return shares, scales


def _counts_from_shares(shares, n_points):
    counts = np.maximum(1, np.round(shares * n_points).astype(int))
    counts[-1] += n_points - counts.sum()
    return np.maximum(counts, 1)


def _generate_blobs(n_points, k, spread, spread_imbalance, rng):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, size=k)
    centers = np.stack([RING_RADIUS * np.cos(angles), RING_RADIUS * np.sin(angles)], axis=1)
    base_std = max(spread, MIN_STD_FRACTION) * RING_RADIUS

    shares, scales = _shares_and_scales(k, spread_imbalance)
    counts = _counts_from_shares(shares, n_points)
    stds = base_std * scales

    points_per_cluster, labels_per_cluster = [], []
    for i in range(k):
        pts = rng.normal(loc=centers[i], scale=stds[i], size=(counts[i], 2))
        points_per_cluster.append(pts)
        labels_per_cluster.append(np.full(counts[i], i))
    return np.concatenate(points_per_cluster, axis=0), np.concatenate(labels_per_cluster, axis=0)


def _generate_moons(n_points, k, spread, spread_imbalance, rng):
    """k=2: klassisches "two moons"-Beispiel (wie sklearn.datasets.make_moons, hier from
    scratch nachgebaut). k>2: k Halbkreis-Boegen als Bluetenblaetter auf einem Ring,
    konkave Seite jeweils zum Ringzentrum (identisches Muster wie dbscan-demo/
    spectral-demo). `spread_imbalance` wirkt wie bei Blobs auf Punktzahl und
    Rauschstaerke von Bogen 0 vs. Bogen 1."""
    shares, scales = _shares_and_scales(k, spread_imbalance)
    counts = _counts_from_shares(shares, n_points)
    base_noise_std = max(spread, MIN_STD_FRACTION) * ARC_RADIUS * 0.3
    noise_stds = base_noise_std * scales

    if k == 2:
        t1 = rng.uniform(0, np.pi, counts[0])
        x1 = ARC_RADIUS * np.cos(t1)
        y1 = ARC_RADIUS * np.sin(t1)

        t2 = rng.uniform(0, np.pi, counts[1])
        x2 = ARC_RADIUS * (1 - np.cos(t2))
        y2 = ARC_RADIUS * (0.5 - np.sin(t2))

        pts1 = np.stack([x1, y1], axis=1) + rng.normal(scale=noise_stds[0], size=(counts[0], 2))
        pts2 = np.stack([x2, y2], axis=1) + rng.normal(scale=noise_stds[1], size=(counts[1], 2))
        points = np.concatenate([pts1, pts2], axis=0)
        labels = np.concatenate([np.zeros(counts[0], dtype=int), np.ones(counts[1], dtype=int)])
        return points, labels

    layout_angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.1, 0.1, size=k)
    arc_centers = np.stack(
        [ARC_RING_RADIUS * np.cos(layout_angles), ARC_RING_RADIUS * np.sin(layout_angles)], axis=1
    )

    points_per_group, labels_per_group = [], []
    for i in range(k):
        t = rng.uniform(0, np.pi, counts[i])
        local_x = ARC_RADIUS * np.cos(t)
        local_y = ARC_RADIUS * np.sin(t)
        rot = layout_angles[i] + np.pi
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        rx = cos_r * local_x - sin_r * local_y
        ry = sin_r * local_x + cos_r * local_y
        pts = np.stack([rx, ry], axis=1) + arc_centers[i] + rng.normal(scale=noise_stds[i], size=(counts[i], 2))
        points_per_group.append(pts)
        labels_per_group.append(np.full(counts[i], i))

    return np.concatenate(points_per_group, axis=0), np.concatenate(labels_per_group, axis=0)


def generate_instance(n_points, k, spread, spread_imbalance, shape, seed):
    rng = np.random.default_rng(seed)
    if shape == "moons":
        points, labels = _generate_moons(n_points, k, spread, spread_imbalance, rng)
    else:
        points, labels = _generate_blobs(n_points, k, spread, spread_imbalance, rng)

    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        shape=shape,
        k=k,
    )
