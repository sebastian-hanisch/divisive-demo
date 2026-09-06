# Divisive Clustering für schrittweise Depot-Aufspaltung – Streamlit-Demo

Neuntes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning". Anders als jedes bisherige Stück dieser Reihe behebt
diese Demo **keine konkrete Schwäche** eines Vorgängers - sie ist ein bewusster
**Kontrast/Vergleich zu [agglomerative-demo](../agglomerative-demo)**: Divisive
Clustering baut - top-down statt bottom-up - dieselbe Art Ergebnis (eine Hierarchie, an
beliebigem k schneidbar) mit einem völlig anderen Mechanismus: **Bisecting k-Means**
statt Lance-Williams-Fusionen.

```
kmeans-demo → dbscan-demo ──┐
                             ├──> hdbscan-demo   (löst BEIDE: über Dichte)
              agglomerative-demo ──────────┘
kmeans-demo → gmm-demo → dpmm-demo             (löst NUR "kein k": über Bayesianische Nichtparametrik)
kmeans-demo → spectral-demo                     (löst NUR Nicht-Konvexität: über Graphentheorie)
kmeans-demo ─┐
             ├──> divisive-demo                 (Kontrast zu agglomerative-demo, KEIN Fix)
agglomerative-demo ─┘
```

Zieht zwei Fäden der Reihe zusammen: den 2-Means-Splitting-Mechanismus aus
[kmeans-demo](../kmeans-demo) und die Hierarchie-/Dendrogramm-Darstellung aus
agglomerative-demo. Erschöpfendes Divisive Clustering müsste pro Schritt alle $2^{n-1}-1$
möglichen Zweiteilungen prüfen - exponentiell viele. **Bisecting k-Means** ersetzt das
durch einen einzigen 2-Means-Aufruf je Split - die einzige in der Praxis gängige
Variante.

## Warum diese Demo anders aufgebaut ist

Weil jeder Split ein 2-Means-Aufruf ist, erbt Bisecting k-Means k-Means'
**Nicht-Konvexitäts-Annahme** - bei Halbmonden scheitert es genau wie k-Means, während
agglomeratives Clustering mit **Single-Linkage** (dieselbe Chaining-Eigenschaft, die bei
agglomerative-demos eigenem "Schwerer Fall"-Preset zur Schwäche wird, ist hier - ohne
Brücke - genau der Mechanismus, der nicht-konvexe Formen sauber trennen kann) weiterhin
erfolgreich ist. Ein ehrlicher Rückbezug: dieselbe Eigenschaft, die dort als Schwäche
(Chaining bei einer Brücke) auftrat, ist hier - bei starker Gruppenüberlappung ganz ohne
Brücke - teils sogar wieder eine Schwäche, während Bisecting k-Means dort robuster
bleibt (siehe Preset 4).

Die Presets legen zwei getrennte Achsen offen:

- **Einfaches Beispiel**: klar getrennte, gleich große/enge Gruppen - Bisecting k-Means
  und die Single-Linkage-Referenz liefern beide eine (fast) perfekte Trennung, kein
  Unterschied nötig.
- **Nicht-konvexe Formen (Halbmonde)**: der Kernnachweis - jeder 2-Means-Split trennt
  nur über eine Hyperebene und scheitert an Halbmonden, während die
  Single-Linkage-Referenz weiterhin sauber trennt.
- **Split-Kriterium macht Unterschied**: ein großer, enger Cluster (viele Punkte,
  niedrige Fehlerquadratsumme) und ein kleiner, diffuser Cluster (wenige Punkte, hohe
  Fehlerquadratsumme) - "Größe" splittet fälschlich den bereits reinen großen Cluster,
  "Fehlerquadratsumme" erkennt korrekt den diffusen als dringlicher.
- **Schwierige Anfangsteilung**: drei stark überlappende Gruppen - schon der erste
  2-Means-Split der Wurzel fällt schwer, und ein suboptimaler früher Split lässt sich
  später nicht mehr korrigieren; bei so starker Überlappung leidet hier sogar die
  Single-Linkage-Referenz stärker (Chaining) als Bisecting k-Means selbst - eine
  ehrliche Erinnerung, dass keins der beiden Verfahren grundsätzlich überlegen ist.

## Visualisierung

Anders als k-Means/EM/Gibbs-Sampling ist Bisecting k-Means **kein iteratives
Verfahren**, sondern ein wiederholter Einzel-Schritt: ein **Split-Schrittregler** zeigt
das **top-down wachsende Dendrogramm** - es entfaltet sich von der Wurzel (ein Cluster,
alle Punkte) nach unten, der direkte visuelle Kontrast zu agglomerative-demos
Blatt-nach-Wurzel-Wachstum, bei sonst identischer Diagrammgrammatik. Das
Dendrogramm-Layout wird einmal über die volle Split-Sequenz (bis zur oberen
Regler-Grenze) berechnet, damit die Blattpositionen beim Durchblättern nicht springen -
dasselbe Prinzip wie agglomerative-demos vorab berechnetes Layout. Ein zweites Diagramm
zeigt die Gesamt-Fehlerquadratsumme gegen die Split-Anzahl (beweisbar monoton fallend).
Kleinmultiples zeigen dieselbe Ziel-Clusteranzahl unter beiden Split-Kriterien
nebeneinander.

## Sicherheitsgrenzen

Keine eigene Iterationsgrenze nötig - das Verfahren terminiert immer nach exakt
$k_{\max}-1$ Splits. `N_POINTS_MAX` (200) hält die naive $O(n^3)$
Single-Linkage-Referenzimplementierung (nur für den Methodenvergleich dieser Demo, siehe
unten) schnell genug für eine flüssige Bedienung - Bisecting k-Means selbst wäre auch bei
deutlich mehr Punkten schnell.

## Verifikation

- **Handgerechnetes Beispiel**: vier Punkte (zwei klar getrennte Paare) von Hand
  durchgerechnet - Fehlerquadratsumme vor und nach dem einzigen sinnvollen Split exakt
  nachgerechnet (101.0 → 1.0).
- **SSE-Monotonie** (Struktur-Invariante, exakt und algorithmusunabhängig): ein
  2-Means-Split eines Clusters kann dessen Fehlerquadratsumme nur senken oder gleich
  lassen - die Gesamt-Fehlerquadratsumme über alle aktiven Cluster ist nach jedem
  weiteren Split beweisbar nicht steigend, live im SSE-Diagramm nachvollziehbar und über
  mehrere Zufallsinstanzen getestet.
- **Kreuzvergleich** gegen `sklearn.cluster.BisectingKMeans`, dessen
  `bisecting_strategy`-Parameter ("largest_cluster"/"biggest_inertia") exakt das eigene
  `split_criterion` ("size"/"sse") abbildet - toleranzbasiert wie bei
  hdbscan-demo/spectral-demo, da k-Means-Interna (Initialisierung, Tie-Breaking)
  abweichen können.
- **Kern-Behauptungen der Demo direkt getestet**: Bisecting k-Means scheitert bei
  Halbmonden, wo die Single-Linkage-Referenz weiterhin sauber trennt
  (`test_bisecting_kmeans_fails_on_moons_where_single_linkage_succeeds`); das
  Split-Kriterium wählt bei der `spread_imbalance`-Szenerie nachweislich unterschiedliche
  Cluster zum Splitten und führt zu einer spürbar unterschiedlichen finalen
  Partitionsqualität (`test_split_criterion_selects_different_parent_cluster_on_diverging_scenario`,
  `test_split_criterion_changes_final_partition_quality`).
- **Alle vier Presets direkt gegen das tatsächliche App-Verhalten getestet**: die
  Primäransicht (Split-Schrittregler) lässt den Algorithmus den Szenario-Seed
  wiederverwenden (etabliertes Muster aus dpmm-demo/spectral-demo), die
  📐-Vergleichssektion nutzt dagegen einen entkoppelten `COMPARISON_SEED` - beide Pfade
  werden geprüft. Dieses Muster deckte während der Entwicklung tatsächlich einen realen
  Tuning-Fehler auf: ein Preset, das nur gegen einen willkürlichen Test-Seed getunt war,
  zeigte in der Primäransicht ein anderes Ergebnis als beabsichtigt - erst durch den
  Abgleich mit dem tatsächlichen App-Verhalten bemerkt, nicht durch einen isolierten Test
  (siehe project-memory für den konkreten Vorfall, dieselbe Fehlerklasse wie bei
  dpmm-demo/spectral-demo).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, Split-Schrittregler mit top-down wachsendem Dendrogramm, Kleinmultiples, Methoden-Vergleich, Formulierungs-Expander |
| `dv_constants.py` | Defaults, Regler-Grenzen, Sicherheitsgrenzen, `PRESETS` |
| `dv_presets.py` | `SettingSpec`/`SETTING_SPECS`, Permalink-Logik, Presets, Zufalls-Seed-Button |
| `dv_scenario.py` | Blobs (mit `spread_imbalance`: ein Cluster wird größer UND enger, ein anderer kleiner UND diffuser) und Halbmonde/Bögen (k>2 als "Blütenblätter auf einem Ring" verallgemeinert wie in dbscan-demo/spectral-demo) |
| `dv_algorithm.py` | Frisches 2-Means (mehrere interne Neustarts), Split-Auswahl nach Kriterium, vollständiges Split-Protokoll, `labels_at_step`, SSE-Monotonie-Hilfsfunktion |
| `dv_evaluation.py` | Rand-Index (from scratch), kompakte Single-Linkage-Referenzimplementierung NUR für den Methodenvergleich (bewusst kein Cross-Import aus agglomerative-demo), Methoden-Vergleich |
| `dv_visualization.py` | Top-down wachsendes Split-Dendrogramm, Gesamt-SSE-über-Splits-Diagramm, Punktwolke, Kleinmultiples, Methoden-Vergleichsdiagramm (Plotly) |
| `tests/` | Handinstanz, SSE-Monotonie, sklearn-Kreuzvergleich, Kern-Nachweise, Preset-gegen-App-Verhalten-Tests, AppTest-Smoke-Test |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
