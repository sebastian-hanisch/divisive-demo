"""Divisive Clustering (Bisecting k-Means) für schrittweise Depot-Aufspaltung -
interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der "Konzepte"-Reihe, gedacht als Kontrast/Vergleich zu
agglomerative-demo (kein Fix einer Schwäche): Divisive Clustering baut - top-down statt
bottom-up - dieselbe Art Hierarchie wie agglomeratives Clustering, mit einem völlig
anderen Mechanismus (Bisecting k-Means statt Lance-Williams-Fusionen). Weil jeder Split
ein 2-Means-Aufruf ist, erbt es k-Means' Nicht-Konvexitäts-Annahme (siehe README für die
DAG-Einordnung).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import dv_constants as C
from dv_algorithm import labels_at_step, run, total_sse_at_step
from dv_evaluation import method_comparison
from dv_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from dv_scenario import generate_instance
from dv_visualization import (
    build_dendrogram_figure,
    build_method_comparison_chart,
    build_mini_scatter_figure,
    build_scatter_figure,
    build_sse_curve_figure,
)

st.set_page_config(page_title="Divisive Clustering – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _compute_run(n_points, k, spread, spread_imbalance, shape, seed, split_criterion):
    instance = generate_instance(n_points, k, spread, spread_imbalance, shape, seed)
    result = run(instance.as_array(), C.TARGET_K_MAX, split_criterion, seed)
    return instance, result


@st.cache_data(show_spinner=False)
def _compute_mini_run(instance, split_criterion, seed):
    return run(instance.as_array(), C.TARGET_K_MAX, split_criterion, seed)


@st.cache_data(show_spinner=False)
def _compute_method_comparison(instance, target_k, split_criterion):
    return method_comparison(instance.as_array(), instance.true_labels, target_k, split_criterion, C.COMPARISON_SEED)


st.title("🌳 Divisive Clustering: schrittweise Depot-Aufspaltung von oben")

st.markdown(
    """
[agglomerative-demo](https://github.com/sebastian-hanisch/agglomerative-demo) baut eine
Hierarchie **von unten**: jeder Punkt startet als eigenes Cluster, Schritt für Schritt wird
das nächstgelegene Paar verschmolzen. **Divisive Clustering** geht den umgekehrten Weg -
**von oben**: alle Standorte starten als EIN großes Verteilzentrum, das wiederholt in zwei
Teile aufgespalten wird, bis die gewünschte Anzahl erreicht ist. Erschöpfendes Divisive
Clustering müsste dafür pro Schritt jede mögliche Zweiteilung prüfen - bei $n$ Punkten
$2^{n-1}-1$ Stück, exponentiell viele. **Bisecting k-Means** ersetzt das durch einen
einzigen 2-Means-Aufruf je Split - die einzige in der Praxis gängige Variante.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren "
    "vergleichen, zeigt diese Demo - Teil der wachsenden \"Konzepte\"-Reihe - **ein** "
    "Verfahren an einem wachsenden Beispiel. Diese Demo behebt dabei bewusst KEINE Schwäche "
    "eines Vorgängers - sie ist ein ehrlicher Kontrast zu agglomerative-demo: dieselbe Art "
    "Hierarchie, entgegengesetzte Bauweise, mit eigenen Kompromissen."
)

with st.expander("So funktioniert Bisecting k-Means", expanded=True):
    st.markdown(
        """
Das Verfahren startet damit, dass alle Punkte EIN Cluster bilden, und wiederholt dann
einen einzigen Schritt, bis die Ziel-Clusteranzahl erreicht ist:

- **Auswahl**: welches der aktuell aktiven Cluster wird gesplittet? Das **Split-Kriterium**
  entscheidet: **Größe** (das Cluster mit den meisten Punkten) oder
  **Fehlerquadratsumme** (das am wenigsten homogene Cluster, gemessen an der Summe der
  quadrierten Abstände zum eigenen Schwerpunkt).
- **Split**: das gewählte Cluster wird mit einem frischen 2-Means (mehrere interne
  Neustarts) in zwei Kinder geteilt.

Jeder Split wird protokolliert - das vollständige Protokoll ist ein **Dendrogramm**, das
sich aber - anders als bei agglomerative-demo - von der **Wurzel nach unten entfaltet**,
nicht von den Blättern nach oben. Die Grafiken weiter unten zeigen genau das live.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = {
    "Einfaches Beispiel": "Klar getrennte, gleich große Gruppen - Bisecting k-Means und die Single-Linkage-Referenz liefern beide eine (fast) perfekte Trennung.",
    "Nicht-konvexe Formen (Halbmonde)": "Der Kernnachweis: jeder 2-Means-Split trennt nur über eine Hyperebene und scheitert an Halbmonden, während die Single-Linkage-Referenz weiterhin sauber trennt.",
    "Split-Kriterium macht Unterschied": "Ein großer, enger Cluster und ein kleiner, diffuser - 'Größe' splittet fälschlich den großen, 'Fehlerquadratsumme' erkennt korrekt den diffusen.",
    "Schwierige Anfangsteilung": "Drei stark überlappende Gruppen - schon der erste Split fällt schwer, und ein suboptimaler früher Split lässt sich später nicht mehr korrigieren.",
}
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_points = st.slider("Anzahl Standorte", *bounds("n_points_slider"), key="n_points_slider")
    k = st.slider("Anzahl wahrer Gruppen", *bounds("k_slider"), key="k_slider")
    spread = st.slider(
        "Streuung", *bounds("spread_slider"), key="spread_slider", step=0.05,
        help="Klein = Gruppen klar getrennt. Groß = Gruppen überlappen sich spürbar.",
    )
    spread_imbalance = st.slider(
        "Größen-/Streuungs-Ungleichgewicht", *bounds("spread_imbalance_slider"),
        key="spread_imbalance_slider", step=0.05,
        help="0 = alle Gruppen gleich groß/eng. Höher = Gruppe 1 wird größer UND enger, "
        "Gruppe 2 kleiner UND diffuser - lässt 'Größe' und 'Fehlerquadratsumme' als "
        "Split-Kriterium auseinanderlaufen.",
    )
    shape = st.radio(
        "Form", options=C.SHAPES, key="shape_radio", format_func=lambda s: C.SHAPE_LABELS[s],
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Bisecting-k-Means-Parameter**")
    split_criterion = st.radio(
        "Split-Kriterium", options=C.SPLIT_CRITERIA, key="split_criterion_radio",
        format_func=lambda c: C.SPLIT_CRITERION_LABELS[c],
        help="Steuert die Primäransicht unten - der '📐'-Vergleich weiter unten nutzt "
        "immer das hier eingestellte Kriterium.",
    )
    target_k = st.slider(
        "Ziel-Clusteranzahl", *bounds("target_k_slider"), key="target_k_slider",
        help="Bei wie vielen Clustern die Primäransicht standardmäßig steht - frei "
        "verschiebbar, um die gesamte Split-Geschichte zu erkunden.",
    )

    st.button(
        "🎲 Neue Punktwolke generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für die Standorte.",
    )

sync_query_params(n_points, k, target_k, spread, spread_imbalance, seed, shape, split_criterion)

with st.spinner("Führe Bisecting k-Means aus..."):
    instance, result = _compute_run(
        int(n_points), int(k), spread, spread_imbalance, shape, int(seed), split_criterion
    )

max_step = result.n_splits - 1
target_step = min(max(int(target_k) - 2, 0), max_step)
run_key = (n_points, k, spread, spread_imbalance, shape, seed, split_criterion, target_k)
if "dv_step" not in st.session_state or st.session_state.get("dv_step_owner") != run_key:
    st.session_state["dv_step"] = target_step
    st.session_state["dv_step_owner"] = run_key

st.markdown("## 🎯 Bisecting k-Means in Aktion")

if max_step == 0:
    step = 0
    st.caption("Nur ein einziger Split möglich - kein Regler nötig.")
else:
    step = st.slider(
        "Schritt (Split)", 0, max_step, key="dv_step",
        help="Ein Schritt = ein Split. Reglerposition entspricht standardmäßig der "
        "eingestellten Ziel-Clusteranzahl - frei verschiebbar, um die gesamte "
        "Split-Geschichte von der Wurzel bis zur höchsten Auflösung zu erkunden.",
    )

dendro_col, scatter_col = st.columns(2)
dendro_col.plotly_chart(
    build_dendrogram_figure(result.splits, step), width="stretch", key=f"dendro_{step}",
)
current_labels = labels_at_step(instance.n_points, result.splits, step)
scatter_col.plotly_chart(
    build_scatter_figure(instance.as_array(), current_labels), width="stretch", key=f"scatter_{step}",
)

lm1, lm2, lm3 = st.columns(3)
lm1.metric("Splits bisher", f"{step + 1}/{max_step + 1}")
lm2.metric("Cluster aktuell", len(set(current_labels)))
lm3.metric(
    "Gesamt-Fehlerquadratsumme", f"{total_sse_at_step(instance.as_array(), instance.n_points, result.splits, step):,.1f}",
    help="Summe der Fehlerquadratsummen aller aktuell aktiven Cluster - nach jedem "
    "weiteren Split beweisbar nicht steigend.",
)

sse_values = [
    total_sse_at_step(instance.as_array(), instance.n_points, result.splits, s)
    for s in range(-1, result.n_splits)
]
st.plotly_chart(build_sse_curve_figure(sse_values), width="stretch", key="sse_curve")

st.markdown("**Und mit anderem Split-Kriterium?**")
st.caption(
    "Gleiche Standorte, gleiche Ziel-Clusteranzahl wie oben - nur das Split-Kriterium "
    "unterscheidet sich."
)
example_cols = st.columns(len(C.SPLIT_CRITERIA))
for col, example_criterion in zip(example_cols, C.SPLIT_CRITERIA):
    with col:
        example_result = _compute_mini_run(instance, example_criterion, int(seed))
        example_step = min(target_step, example_result.n_splits - 1)
        example_labels = labels_at_step(instance.n_points, example_result.splits, example_step)
        st.plotly_chart(
            build_mini_scatter_figure(instance.as_array(), example_labels),
            width="stretch", key=f"mini_{example_criterion}",
        )
        st.caption(C.SPLIT_CRITERION_LABELS[example_criterion])

st.markdown("---")

st.subheader("📐 Was kann Bisecting k-Means, was Single-Linkage nicht kann - und was nicht?")
st.markdown(
    """
Live für Ihr aktuelles Szenario berechnet, nicht nur behauptet: der **Rand-Index** (Anteil
der Punktpaare, bei denen die berechnete Aufteilung mit der tatsächlichen
Gruppenzugehörigkeit übereinstimmt - 1.0 = perfekte Übereinstimmung, ~0.5 = kaum besser als
Zufall) für Bisecting k-Means (beim eingestellten Split-Kriterium) gegen eine
Single-Linkage-Referenz, beide beim gleichen Ziel-k:
"""
)

scores = _compute_method_comparison(instance, int(target_k), split_criterion)
st.plotly_chart(build_method_comparison_chart(scores), width="stretch", key="comparison_bar")

convexity_gap = scores["single_linkage_baseline"] - scores["bisecting"]
if convexity_gap > 0.2 and shape == "moons":
    st.success(
        f"✅ Bei diesem Szenario erreicht Bisecting k-Means nur einen Rand-Index von "
        f"{scores['bisecting']:.2f}, während die Single-Linkage-Referenz bei "
        f"{scores['single_linkage_baseline']:.2f} liegt - genau die Nicht-Konvexitäts-"
        f"Annahme, die Bisecting k-Means über seinen 2-Means-Splitting-Mechanismus von "
        f"k-Means erbt (dieselbe Chaining-Eigenschaft, die bei agglomerative-demos "
        f"Brücken-Preset zur Schwäche wird, ist hier - ohne Brücke - der Grund für den "
        f"Erfolg der Referenz)."
    )
elif convexity_gap > 0.15 and shape == "blobs":
    st.warning(
        f"⚠️ Bei diesem Szenario erreicht Bisecting k-Means nur einen Rand-Index von "
        f"{scores['bisecting']:.2f}, obwohl die Gruppen konvex sind - hier liegt es NICHT "
        f"an der Form, sondern am **Split-Kriterium**: probieren Sie in der Seitenleiste "
        f"das jeweils andere Kriterium aus, um zu sehen, ob sich das Ergebnis verbessert "
        f"(siehe auch die Kleinmultiples oben, die beide Kriterien direkt "
        f"gegenüberstellen)."
    )
elif scores["bisecting"] - scores["single_linkage_baseline"] > 0.2:
    st.info(
        f"ℹ️ Hier schneidet Bisecting k-Means ({scores['bisecting']:.2f}) deutlich besser "
        f"ab als die Single-Linkage-Referenz ({scores['single_linkage_baseline']:.2f}) - "
        f"bei so starker Überlappung leidet Single-Linkage an **Chaining** (siehe "
        f"agglomerative-demo), während Bisecting k-Means' Hyperebenen-Splits hier robuster "
        f"sind. Kein Verfahren ist grundsätzlich überlegen - welches besser passt, hängt "
        f"von der Datenstruktur ab."
    )
elif scores["bisecting"] < 0.85 and scores["single_linkage_baseline"] < 0.85:
    st.info(
        f"ℹ️ Beide Verfahren tun sich bei diesem Szenario schwer (Bisecting k-Means: "
        f"{scores['bisecting']:.2f}, Single-Linkage-Referenz: "
        f"{scores['single_linkage_baseline']:.2f}) - aus unterschiedlichen Gründen: "
        f"Bisecting k-Means committet sich früh auf einen 2-Means-Split, der sich nicht "
        f"mehr korrigieren lässt; Single-Linkage leidet bei starker Überlappung an "
        f"Chaining."
    )
else:
    st.info(
        "Bei diesem Szenario liegen beide Verfahren nah beieinander - keiner der beiden "
        "Mechanismen (Hyperebenen-Split vs. Chaining-Anfälligkeit) wird hier auf die Probe "
        "gestellt."
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Algorithmus (Bisecting k-Means)**: starte mit einem Cluster $C_0$ = alle Punkte.
Wiederhole, bis $k$ Cluster aktiv sind:

1. Wähle das zu splittende Cluster $C$ nach dem Split-Kriterium - **Größe**: $\arg\max_C
   |C|$, oder **Fehlerquadratsumme**: $\arg\max_C \text{SSE}(C)$ mit
   $\text{SSE}(C) = \sum_{x \in C} \lVert x - \bar{x}_C \rVert^2$.
2. Splitte $C$ mit 2-Means (mehrere Neustarts, behalte den Lauf mit niedrigster
   End-SSE) in zwei Kinder $C_a, C_b$.

**Warum nicht erschöpfend?** Der naive divisive Ansatz würde bei jedem Schritt ALLE
möglichen Zweiteilungen eines Clusters mit $m$ Punkten prüfen - $2^{m-1}-1$ Stück,
exponentiell. Bisecting k-Means ersetzt diese erschöpfende Suche durch einen einzigen
2-Means-Aufruf: eine Näherung, keine exakte Lösung.

**SSE-Monotonie**: ein 2-Means-Split eines Clusters $C$ in $C_a, C_b$ kann dessen Beitrag
zur Gesamt-SSE nur senken oder gleich lassen, da der Schwerpunkt jedes Kindes näher an
seinen eigenen Punkten liegt als der ursprüngliche gemeinsame Schwerpunkt:

$$
\text{SSE}(C_a) + \text{SSE}(C_b) \le \text{SSE}(C)
$$

Die Gesamt-SSE über alle aktiven Cluster ist damit nach jedem weiteren Split beweisbar
nicht steigend - live im SSE-Diagramm oben nachvollziehbar.

**Rand-Index**: Anteil der Punktpaare, bei denen zwei Partitionen übereinstimmen (beide
sagen "gleiches Cluster" oder beide sagen "unterschiedliches Cluster"):

$$
\text{RI} = \frac{a + b}{\binom{n}{2}}
$$

wobei $a$ die Anzahl der Paare ist, die in beiden Partitionen im selben Cluster liegen, und
$b$ die Anzahl der Paare, die in beiden Partitionen in unterschiedlichen Clustern liegen.

**Ehrlicher Kompromiss**: Bisecting k-Means braucht - wie k-Means selbst - weiterhin eine
vorab feststehende Ziel-Clusteranzahl $k$, und jeder Split ist (wie bei k-Means) durch eine
Hyperebene begrenzt - nicht-konvexe Formen bleiben ein Problem. Was es dafür bietet:
deutlich günstiger als eine volle $O(n^2)$-/$O(n^3)$-Abstandsmatrix-Berechnung wie bei
agglomerative-demo, und trotzdem eine vollständige, schneidbare Hierarchie.

Implementiert in `dv_algorithm.py` (Bisecting k-Means) und `dv_evaluation.py`
(Rand-Index, Single-Linkage-Referenz, Methodenvergleich).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Clustering erklärt: k-Means bis HDBSCAN](https://sebastianhanisch.net/konzepte-clustering.html)."
)
