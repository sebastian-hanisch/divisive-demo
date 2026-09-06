import numpy as np

from dv_scenario import generate_instance


def test_point_count_matches_request():
    instance = generate_instance(n_points=100, k=3, spread=0.2, spread_imbalance=0.0, shape="blobs", seed=1)
    assert instance.n_points == 100


def test_reproducible_given_same_seed():
    kwargs = dict(n_points=80, k=3, spread=0.2, spread_imbalance=0.4, shape="blobs", seed=42)
    a = generate_instance(**kwargs)
    b = generate_instance(**kwargs)
    assert a.points == b.points
    assert a.true_labels == b.true_labels


def test_moons_k_equals_two_produces_two_true_groups():
    instance = generate_instance(n_points=150, k=2, spread=0.08, spread_imbalance=0.0, shape="moons", seed=2)
    labels = np.array(instance.true_labels)
    assert set(labels.tolist()) == {0, 1}


def test_moons_generalizes_to_k_greater_than_two():
    instance = generate_instance(n_points=150, k=4, spread=0.08, spread_imbalance=0.0, shape="moons", seed=2)
    labels = np.array(instance.true_labels)
    assert set(labels.tolist()) == {0, 1, 2, 3}


def test_spread_imbalance_makes_group_zero_bigger_and_tighter_under_blobs():
    balanced = generate_instance(n_points=150, k=3, spread=0.2, spread_imbalance=0.0, shape="blobs", seed=3)
    imbalanced = generate_instance(n_points=150, k=3, spread=0.2, spread_imbalance=0.9, shape="blobs", seed=3)

    def group_stats(instance):
        points = np.array(instance.points)
        labels = np.array(instance.true_labels)
        counts = np.bincount(labels, minlength=3)
        std0 = points[labels == 0].std()
        std1 = points[labels == 1].std()
        return counts, std0, std1

    counts_b, std0_b, std1_b = group_stats(balanced)
    counts_i, std0_i, std1_i = group_stats(imbalanced)

    assert counts_i[0] > counts_b[0]
    assert counts_i[1] < counts_b[1]
    assert std0_i < std0_b
    assert std1_i > std1_b


def test_spread_imbalance_also_affects_moons():
    balanced = generate_instance(n_points=150, k=3, spread=0.08, spread_imbalance=0.0, shape="moons", seed=4)
    imbalanced = generate_instance(n_points=150, k=3, spread=0.08, spread_imbalance=0.9, shape="moons", seed=4)
    counts_b = np.bincount(np.array(balanced.true_labels), minlength=3)
    counts_i = np.bincount(np.array(imbalanced.true_labels), minlength=3)
    assert counts_i[0] > counts_b[0]
    assert counts_i[1] < counts_b[1]
