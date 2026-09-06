import pytest

import dv_constants as C
from dv_algorithm import labels_at_step, run
from dv_evaluation import method_comparison, rand_index, single_linkage_labels
from dv_scenario import generate_instance


def test_rand_index_is_one_for_identical_partitions():
    assert rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_rand_index_is_one_up_to_relabeling():
    assert rand_index([0, 0, 1, 1], [5, 5, 9, 9]) == pytest.approx(1.0)


def test_rand_index_penalizes_disagreement():
    assert rand_index([0, 0, 1, 1], [0, 1, 0, 1]) < 0.6


def test_single_linkage_baseline_returns_k_distinct_labels():
    instance = generate_instance(n_points=60, k=3, spread=0.15, spread_imbalance=0.0, shape="blobs", seed=1)
    labels = single_linkage_labels(instance.as_array(), k=3)
    assert len(set(labels.tolist())) == 3


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bisecting_kmeans_fails_on_moons_where_single_linkage_succeeds(seed):
    """Kern-Nachweis 1: bei nicht-konvexen Halbmonden trennt die Single-Linkage-
    Referenz sauber, waehrend Bisecting k-Means (jeder Split ist ein 2-Means-Aufruf,
    also eine Hyperebenen-Trennung) nachweislich scheitert - dieselbe
    Konvexitaets-Annahme wie bei plain k-Means, hier nur ueber den Split-Mechanismus
    geerbt."""
    instance = generate_instance(n_points=150, k=2, spread=0.08, spread_imbalance=0.0, shape="moons", seed=seed)
    scores = method_comparison(instance.as_array(), instance.true_labels, target_k=2, split_criterion="sse", seed=1)
    assert scores["single_linkage_baseline"] > 0.9
    assert scores["bisecting"] < 0.75


def test_both_methods_fine_on_convex_blobs():
    """Baseline-Gegenprobe: bei konvexen, klar getrennten Gruppen gibt es keinen
    Vorteil - beide Verfahren schneiden etwa gleich gut ab."""
    instance = generate_instance(n_points=90, k=3, spread=0.15, spread_imbalance=0.0, shape="blobs", seed=1)
    scores = method_comparison(instance.as_array(), instance.true_labels, target_k=3, split_criterion="sse", seed=1)
    assert scores["bisecting"] > 0.9
    assert scores["single_linkage_baseline"] > 0.9


def test_split_criterion_changes_final_partition_quality():
    """Kern-Nachweis 2: bei der `spread_imbalance`-Szenerie liefert "size" eine
    spuerbar schlechtere finale Partition als "sse", weil es den bereits reinen,
    grossen Cluster splittet statt den kleinen, diffusen."""
    instance = generate_instance(n_points=120, k=3, spread=0.15, spread_imbalance=0.85, shape="blobs", seed=7)
    scores_size = method_comparison(instance.as_array(), instance.true_labels, target_k=3, split_criterion="size", seed=1)
    scores_sse = method_comparison(instance.as_array(), instance.true_labels, target_k=3, split_criterion="sse", seed=1)
    assert scores_sse["bisecting"] > 0.9
    assert scores_sse["bisecting"] - scores_size["bisecting"] > 0.15


def _run_preset_as_app_would(preset_name):
    """Repliziert app.py's tatsaechliches Verhalten exakt: die PRIMAERANSICHT
    (Split-Schrittregler) laesst `run()` den Szenario-Seed wiederverwenden (etabliertes
    Muster aus dpmm-demo/spectral-demo), die 📐-Vergleichssektion nutzt dagegen den
    entkoppelten `COMPARISON_SEED`. Beide Pfade werden hier geprueft - Lehre aus
    spectral-demo, wo ein Preset nur gegen einen willkuerlichen Test-Seed getunt war und
    live ein anderes Ergebnis zeigte."""
    p = C.PRESETS[preset_name]
    instance = generate_instance(p["n_points"], p["k"], p["spread"], p["spread_imbalance"], p["shape"], p["seed"])
    data = instance.as_array()

    primary_result = run(data, p["target_k"], p["split_criterion"], p["seed"])
    primary_labels = labels_at_step(p["n_points"], primary_result.splits, p["target_k"] - 2)
    primary_ri = rand_index(instance.true_labels, primary_labels)

    comparison_scores = method_comparison(
        data, instance.true_labels, p["target_k"], p["split_criterion"], C.COMPARISON_SEED
    )
    return instance, p, primary_ri, comparison_scores


def test_simple_preset_matches_actual_app_behavior():
    _, _, primary_ri, scores = _run_preset_as_app_would("Einfaches Beispiel")
    assert primary_ri > 0.95
    assert scores["bisecting"] > 0.95
    assert scores["single_linkage_baseline"] > 0.95


def test_moons_preset_matches_actual_app_behavior():
    _, _, primary_ri, scores = _run_preset_as_app_would("Nicht-konvexe Formen (Halbmonde)")
    assert primary_ri < 0.75
    assert scores["bisecting"] < 0.75
    assert scores["single_linkage_baseline"] > 0.9


def test_split_criterion_preset_actually_uses_a_bad_default_criterion():
    """Die eigentliche Regression: das Preset muss standardmaessig "size" (das
    schwaechere Kriterium fuer diese Szenerie) einstellen, sonst zeigt es die
    Divergenz nicht beim Laden."""
    p = C.PRESETS["Split-Kriterium macht Unterschied"]
    assert p["split_criterion"] == "size"


def test_split_criterion_preset_matches_actual_app_behavior():
    _, p, primary_ri, scores = _run_preset_as_app_would("Split-Kriterium macht Unterschied")
    assert p["split_criterion"] == "size"
    assert primary_ri < 0.85
    assert scores["bisecting"] < 0.85

    switched = method_comparison(
        generate_instance(p["n_points"], p["k"], p["spread"], p["spread_imbalance"], p["shape"], p["seed"]).as_array(),
        generate_instance(p["n_points"], p["k"], p["spread"], p["spread_imbalance"], p["shape"], p["seed"]).true_labels,
        p["target_k"], "sse", C.COMPARISON_SEED,
    )
    assert switched["bisecting"] > 0.95


def test_hard_root_split_preset_matches_actual_app_behavior():
    _, _, primary_ri, _ = _run_preset_as_app_would("Schwierige Anfangsteilung")
    assert primary_ri < 0.9
