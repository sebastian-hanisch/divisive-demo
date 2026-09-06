import numpy as np
import pytest

from dv_algorithm import labels_at_step, run, total_sse_at_step
from dv_scenario import generate_instance


def test_hand_computed_two_pairs_split():
    """Vier Punkte, von Hand konstruiert: A=(0,0), B=(1,0) bilden ein enges Paar bei
    x~0.5, C=(10,0), D=(11,0) ein zweites bei x~10.5 - weit getrennt von {A,B}. Der
    einzige sinnvolle 2-Means-Split der Wurzel muss {A,B} von {C,D} trennen. Von Hand
    nachgerechnet: Gesamtschwerpunkt (5.5, 0), SSE davor = (5.5^2+4.5^2)*2 = 101.0;
    nach dem Split haben beide Paare Schwerpunkt-Abstand 0.5 je Punkt, SSE danach =
    (0.5^2+0.5^2)*2 = 1.0."""
    data = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
    result = run(data, max_k=2, split_criterion="sse", seed=1)

    assert result.n_splits == 1
    split = result.splits[0]
    assert split.sse_before == pytest.approx(101.0)
    assert split.sse_after == pytest.approx(1.0)

    groups = {frozenset(split.members_a), frozenset(split.members_b)}
    assert groups == {frozenset({0, 1}), frozenset({2, 3})}


def test_exactly_max_k_minus_one_splits_and_growing_cluster_count():
    instance = generate_instance(n_points=60, k=3, spread=0.15, spread_imbalance=0.0, shape="blobs", seed=1)
    max_k = 6
    result = run(instance.as_array(), max_k, "sse", seed=1)
    assert result.n_splits == max_k - 1
    for step in range(result.n_splits):
        n_clusters = len(set(labels_at_step(instance.n_points, result.splits, step)))
        assert n_clusters == step + 2  # step=0 -> 2 Cluster, step=1 -> 3, ...


def test_total_sse_never_increases_after_a_split():
    """Struktur-Invariante: ein 2-Means-Split kann die Fehlerquadratsumme eines Clusters
    nur senken oder gleich lassen - die Gesamt-SSE ueber alle aktiven Cluster ist also
    nach jedem weiteren Split monoton nicht steigend, analog zu kmeans-demos
    Inertia-Monotonie ueber Iterationen."""
    for seed in range(5):
        instance = generate_instance(n_points=70, k=3, spread=0.3, spread_imbalance=0.3, shape="blobs", seed=seed)
        data = instance.as_array()
        result = run(data, max_k=6, split_criterion="sse", seed=seed)
        sse_values = [total_sse_at_step(data, instance.n_points, result.splits, step) for step in range(-1, result.n_splits)]
        for i in range(len(sse_values) - 1):
            assert sse_values[i + 1] <= sse_values[i] + 1e-6


@pytest.mark.parametrize("criterion,bisecting_strategy", [("size", "largest_cluster"), ("sse", "biggest_inertia")])
def test_matches_sklearn_bisecting_kmeans_partition(criterion, bisecting_strategy):
    """Unabhaengiger Kreuzvergleich gegen sklearn.cluster.BisectingKMeans, dessen
    `bisecting_strategy`-Parameter ("largest_cluster"/"biggest_inertia") exakt unser
    `split_criterion` ("size"/"sse") abbildet. Toleranzbasiert (wie bei hdbscan-demo/
    spectral-demo), da k-Means-Interna (Initialisierung, Tie-Breaking) abweichen koennen -
    sklearn ist ausschliesslich ein Test-Dependency, kein Laufzeit-Dependency der App."""
    sklearn_cluster = pytest.importorskip("sklearn.cluster")

    for seed in range(1, 4):
        instance = generate_instance(n_points=90, k=3, spread=0.15, spread_imbalance=0.0, shape="blobs", seed=seed)
        data = instance.as_array()
        result = run(data, max_k=3, split_criterion=criterion, seed=seed)
        ours = np.array(labels_at_step(instance.n_points, result.splits, 1))

        theirs = sklearn_cluster.BisectingKMeans(
            n_clusters=3, bisecting_strategy=bisecting_strategy, random_state=seed, n_init=5
        ).fit(data).labels_

        n = len(data)
        iu = np.triu_indices(n, k=1)
        same_ours = (ours[:, None] == ours[None, :])[iu]
        same_theirs = (theirs[:, None] == theirs[None, :])[iu]
        agreement = float(np.mean(same_ours == same_theirs))
        assert agreement > 0.85, f"seed {seed}, {criterion}: agreement {agreement}"


def test_split_criterion_selects_different_parent_cluster_on_diverging_scenario():
    """Kern-Nachweis fuer das eine method-eigene Kriterium dieser Demo: bei der
    `spread_imbalance`-Szenerie waehlen "size" und "sse" beim zweiten Split (Schritt 1)
    nachweislich unterschiedliche Cluster zum Splitten - "size" splittet faelschlich
    den grossen, bereits engen Cluster, "sse" erkennt korrekt den kleinen, diffusen."""
    instance = generate_instance(n_points=120, k=3, spread=0.15, spread_imbalance=0.85, shape="blobs", seed=7)
    data = instance.as_array()

    by_size = run(data, max_k=3, split_criterion="size", seed=7)
    by_sse = run(data, max_k=3, split_criterion="sse", seed=7)

    assert by_size.splits[1].parent_cluster != by_sse.splits[1].parent_cluster
