"""Plotly-Visualisierungen: Punktwolke, das top-down wachsende Split-Dendrogramm (neu
in dieser Reihe - waechst von der Wurzel nach unten, Kontrast zu agglomerative-demos
Blatt-nach-Wurzel-Wachstum), Gesamt-SSE-über-Splits-Diagramm, Kleinmultiples (beide
Split-Kriterien) und Methoden-Vergleichsdiagramm."""

import numpy as np

CLUSTER_PALETTE = [
    "#1f77b4", "#d68a2e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _cluster_color(index, n_clusters):
    if n_clusters <= len(CLUSTER_PALETTE):
        return CLUSTER_PALETTE[index % len(CLUSTER_PALETTE)]
    hue = (index * 360.0 / n_clusters) % 360
    return f"hsl({hue:.1f}, 65%, 50%)"


def _axis_range(data):
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    padx = (xmax - xmin) * 0.1 or 1.0
    pady = (ymax - ymin) * 0.1 or 1.0
    return [xmin - padx, xmax + padx], [ymin - pady, ymax + pady]


def _cluster_traces(data, labels, legend):
    import plotly.graph_objects as go

    traces = []
    cluster_ids = sorted(set(labels.tolist()))
    n_clusters = len(cluster_ids)
    show_legend = legend and n_clusters <= 12
    for index, cid in enumerate(cluster_ids):
        mask = labels == cid
        color = _cluster_color(index, n_clusters)
        traces.append(
            go.Scatter(
                x=data[mask, 0], y=data[mask, 1], mode="markers", name=f"Cluster {cid + 1}",
                showlegend=show_legend,
                marker=dict(color=color, size=7, line=dict(width=0.5, color="white")),
                hoverinfo="skip",
            )
        )
    return traces


def _build_scatter(data, labels, height, legend):
    import plotly.graph_objects as go

    fig = go.Figure()
    for trace in _cluster_traces(data, np.asarray(labels), legend):
        fig.add_trace(trace)

    xr, yr = _axis_range(data)
    layout_kwargs = dict(
        template="plotly_white", height=height,
        xaxis=dict(visible=False, range=xr, fixedrange=True),
        yaxis=dict(visible=False, range=yr, fixedrange=True, scaleanchor="x", scaleratio=1),
        showlegend=legend,
        margin=dict(t=40 if legend else 5, l=10 if legend else 5, r=10 if legend else 5, b=10 if legend else 5),
    )
    if legend:
        layout_kwargs["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(**layout_kwargs)
    return fig


def build_scatter_figure(data, labels):
    return _build_scatter(data, labels, height=460, legend=True)


def build_mini_scatter_figure(data, labels):
    return _build_scatter(data, labels, height=220, legend=False)


def _leaf_order_and_children(splits):
    children = {s.parent_cluster: (s.child_a, s.child_b) for s in splits}
    order = []

    def visit(cid):
        if cid in children:
            a, b = children[cid]
            visit(a)
            visit(b)
        else:
            order.append(cid)

    visit(0)
    return order, children


def _x_positions(order, children):
    cache = {cid: float(i) for i, cid in enumerate(order)}

    def x_of(cid):
        if cid in cache:
            return cache[cid]
        a, b = children[cid]
        val = (x_of(a) + x_of(b)) / 2.0
        cache[cid] = val
        return val

    for cid in children:
        x_of(cid)
    x_of(0)
    return cache


def _creation_y(cid, splits):
    if cid == 0:
        return 0.0
    for s in splits:
        if cid in (s.child_a, s.child_b):
            return -(s.step + 1)
    return 0.0


def _active_clusters(splits, step):
    clusters = {0}
    for s in splits[: step + 1]:
        clusters.discard(s.parent_cluster)
        clusters.add(s.child_a)
        clusters.add(s.child_b)
    return clusters


def build_dendrogram_figure(splits, step):
    """Zeichnet den Split-Baum von der Wurzel (Schritt -1, ein Cluster) nach unten - der
    direkte visuelle Kontrast zu agglomerative-demos Blatt-nach-Wurzel-Dendrogramm. Das
    x-Layout wird einmal über die VOLLE Split-Sequenz (bis max_k) berechnet, damit die
    Blattpositionen beim Durchblättern des Schrittreglers nicht springen - exakt das
    Prinzip aus agglomerative-demos vorab berechnetem Dendrogramm-Layout. Vereinfachte
    Darstellung mit geraden (nicht rechtwinkligen) Verbindungslinien."""
    import plotly.graph_objects as go

    order, children = _leaf_order_and_children(splits)
    x_of = _x_positions(order, children)

    fig = go.Figure()
    for s in splits[: step + 1]:
        py = -s.step
        cy = -(s.step + 1)
        px = x_of[s.parent_cluster]
        for child in (s.child_a, s.child_b):
            fig.add_trace(
                go.Scatter(
                    x=[px, x_of[child]], y=[py, cy], mode="lines",
                    line=dict(color="#9aa6ba", width=2), hoverinfo="skip", showlegend=False,
                )
            )

    bottom_y = -(step + 1)
    for cid in _active_clusters(splits, step):
        cy = _creation_y(cid, splits)
        if cy > bottom_y:
            fig.add_trace(
                go.Scatter(
                    x=[x_of[cid], x_of[cid]], y=[cy, bottom_y], mode="lines",
                    line=dict(color="#c7ceda", width=2, dash="dot"), hoverinfo="skip", showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[x_of[cid]], y=[bottom_y], mode="markers",
                marker=dict(size=8, color="#1f77b4"), hoverinfo="skip", showlegend=False,
            )
        )

    max_depth = len(splits)
    fig.update_layout(
        template="plotly_white", height=340,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(
            title="Split-Tiefe", fixedrange=True,
            tickmode="array",
            tickvals=list(range(0, -(max_depth + 1), -1)),
            ticktext=[str(-v) for v in range(0, -(max_depth + 1), -1)],
        ),
        margin=dict(t=10, l=10, r=10, b=10), showlegend=False,
    )
    return fig


def build_sse_curve_figure(sse_values):
    """Gesamt-SSE gegen Anzahl Splits - beweisbar monoton nicht steigend (siehe
    dv_algorithm.total_sse_at_step), Analogon zu kmeans-demos Inertia-Kurve."""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=list(range(len(sse_values))), y=sse_values, mode="lines+markers",
            line=dict(color="#1f77b4"),
        )
    )
    fig.update_layout(
        template="plotly_white", height=260,
        xaxis=dict(title="Anzahl Splits", fixedrange=True, dtick=1),
        yaxis=dict(title="Gesamt-Fehlerquadratsumme", fixedrange=True),
        margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig


def build_method_comparison_chart(scores):
    import plotly.graph_objects as go

    labels = ["Bisecting k-Means", "Single-Linkage-Referenz"]
    values = [scores["bisecting"], scores["single_linkage_baseline"]]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=["#1f77b4", "#2ca02c"]))
    fig.update_layout(
        template="plotly_white", height=280,
        yaxis=dict(title="Rand-Index", range=[0, 1.05], fixedrange=True),
        xaxis=dict(fixedrange=True), margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig
