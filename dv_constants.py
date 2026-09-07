"""Defaults, Regler-Grenzen, Sicherheitsgrenzen und Presets für die Divisive-Clustering-
(Bisecting-k-Means-)Demo."""

DEFAULT_N_POINTS = 90
DEFAULT_K = 3
DEFAULT_TARGET_K = 3
DEFAULT_SPREAD = 0.15
DEFAULT_SPREAD_IMBALANCE = 0.0
DEFAULT_SEED = 1
DEFAULT_SHAPE = "blobs"
DEFAULT_SPLIT_CRITERION = "sse"

N_POINTS_MIN, N_POINTS_MAX = 30, 200
K_MIN, K_MAX = 2, 6
TARGET_K_MIN, TARGET_K_MAX = 2, 8
SPREAD_MIN, SPREAD_MAX = 0.05, 0.9
SPREAD_IMBALANCE_MIN, SPREAD_IMBALANCE_MAX = 0.0, 1.0

SHAPES = ("blobs", "moons")
SHAPE_LABELS = {"blobs": "Gruppen (Blobs)", "moons": "Halbmonde"}

SPLIT_CRITERIA = ("size", "sse")
SPLIT_CRITERION_LABELS = {
    "size": "Größe (meiste Punkte)",
    "sse": "Fehlerquadratsumme (am wenigsten homogen)",
}

# Sicherheitsgrenze - die Single-Linkage-Referenzimplementierung fuer den
# Methodenvergleich ist bewusst naiv O(n^3), Bisecting k-Means selbst waere auch bei
# deutlich mehr Punkten schnell.
N_POINTS_HARD_MAX = 200

# Feste interne Neustart-Anzahl je 2-Means-Split (nicht als Regler exponiert - reduziert,
# aber beseitigt nicht das Lokale-Optima-Risiko aus kmeans-demo).
N_RESTARTS = 5

# Fester Vergleichs-Seed fuer den Methodenvergleich, unabhaengig vom Szenario-Seed
# (Lehre aus gmm-demo/dpmm-demo/spectral-demo: Vergleichs-Randomness nie an den
# Szenario-Seed koppeln).
COMPARISON_SEED = 1

PRESETS = {
    "Einfaches Beispiel": {
        "n_points": 90, "k": 3, "target_k": 3, "spread": 0.15, "spread_imbalance": 0.0,
        "shape": "blobs", "split_criterion": "sse", "seed": 1,
    },
    "Nicht-konvexe Formen (Halbmonde)": {
        "n_points": 150, "k": 2, "target_k": 2, "spread": 0.08, "spread_imbalance": 0.0,
        "shape": "moons", "split_criterion": "sse", "seed": 2,
    },
    "Split-Kriterium macht Unterschied": {
        "n_points": 120, "k": 3, "target_k": 3, "spread": 0.15, "spread_imbalance": 0.85,
        "shape": "blobs", "split_criterion": "size", "seed": 7,
    },
    "Schwierige Anfangsteilung": {
        "n_points": 100, "k": 3, "target_k": 3, "spread": 0.45, "spread_imbalance": 0.0,
        "shape": "blobs", "split_criterion": "sse", "seed": 26,
    },
}
