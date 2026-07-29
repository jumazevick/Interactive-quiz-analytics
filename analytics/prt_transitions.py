from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_NODE_RE = re.compile(r"-(\d+)-([TF])$")

# Green (few students traversed this edge) -> yellow -> red (many students traversed
# it); the colorblind variant swaps in the Okabe-Ito blue/yellow/vermillion triple used
# elsewhere in this app for red-green-unsafe scales (see analytics/ui_theme.py), kept
# local here since the semantic direction (low->high traffic) is specific to this chart.
_DEFAULT_TRAFFIC_SCALE = ["#22c55e", "#fde68a", "#ef4444"]
_COLORBLIND_TRAFFIC_SCALE = ["#0072B2", "#F0E442", "#D55E00"]


def get_part(row: "pd.Series", part_index: int = 1) -> dict | None:
    """The PRT dict for one part of a multi-part question (`prt1` -> part_index 1,
    `prt2` -> 2, ...), or None if this response has no such part."""
    for prt in (row.get("prt_list") or []):
        if prt.get("index") == part_index:
            return prt
    return None


def count_question_parts(response_df: pd.DataFrame, question: str) -> int:
    """How many PRT parts (prt1, prt2, ...) this question has, taken as the largest part
    index seen across every response to it — a blank/invalid response can omit trailing
    parts, so the maximum rather than the first row's count is what's authoritative."""
    rows = response_df[response_df["question"] == question]
    max_index = 0
    for prt_list in rows.get("prt_list", pd.Series(dtype=object)):
        for prt in (prt_list or []):
            max_index = max(max_index, int(prt.get("index") or 0))
    return max(max_index, 1)


def _terminal_node(part: dict) -> tuple[int, str] | None:
    """(node_number, "T"|"F") for the node a PRT traversal ended at, or None if this
    part carries no usable trace (blank/invalid input)."""
    notes = part.get("answer_notes") or []
    if not notes:
        single = str(part.get("answer_note") or "")
        notes = [single] if single else []
    for note in reversed(notes):
        match = _NODE_RE.search(str(note).strip())
        if match:
            return int(match.group(1)), match.group(2)
    return None


def classify_node(row: "pd.Series", part_index: int = 1) -> str:
    """Classify one response to one part of a question into a transition-graph node
    label, reproducing the mapping in the reference implementation (network_analysis.py,
    which derives `answer_number` from whichever PRT node evaluated True and then applies
    `mapping = {0: 0, 1: 'c', 2: 1, 3: 2, 4: 3, 5: 4}`):

    * full marks on this part -> `"c"` (node 1 True in that script's PRT design)
    * traversal ended True at node k>1 -> `str(k - 1)`, so node 2 -> "1", node 3 -> "2"
    * traversal ended False, or the part is missing/blank/invalid -> `"0"` (unclassified)

    Correctness is judged per part (`fraction == 1.0`) rather than from the row's overall
    `response_status`, which on a multi-part question only says "every part was right" —
    that masked a student who nailed *this* part while missing another one.
    """
    part = get_part(row, part_index)
    if part is None:
        return "0"

    if part.get("fraction") == 1.0:
        return "c"

    terminal = _terminal_node(part)
    if terminal is None:
        return "0"

    node_number, outcome = terminal
    if outcome != "T":
        # Fell through the last node without matching it — the PRT made no
        # classification at all, so this is an unclassified wrong answer, not "node k".
        return "0"
    return "c" if node_number == 1 else str(node_number - 1)


def node_sort_key(node: str) -> tuple[int, int, str]:
    """Deterministic node ordering: `"0"` first, ascending numeric nodes next, `"c"`
    last — used both for layout and for any node-ordered display."""
    if node == "0":
        return (0, 0, node)
    if node == "c":
        return (2, 0, node)
    try:
        return (1, int(node), node)
    except ValueError:
        return (1, 10**6, node)


def build_student_node_sequence(
    response_df: pd.DataFrame,
    question: str,
    student_id: str,
    part_index: int = 1,
) -> pd.DataFrame:
    """Ordered (by `completed_dt`, then `attempt_idx`) node-classification sequence for
    one student's attempts at one part of one question. `response_df` should be Pool A
    (all attempts, not just each student's best) so a retry trajectory is visible."""
    columns = ["attempt_order", "node", "grade", "completed_dt", "attempt_idx"]
    rows = response_df[(response_df["question"] == question) & (response_df["student_id"] == student_id)]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    rows = rows.sort_values(by=["completed_dt", "attempt_idx"])
    out = rows[["grade", "completed_dt", "attempt_idx"]].copy()
    out["node"] = rows.apply(classify_node, axis=1, part_index=part_index)
    out = out.reset_index(drop=True)
    out.insert(0, "attempt_order", range(len(out)))
    return out[columns]


def build_transition_pairs(nodes: list[str]) -> list[tuple[str, str]]:
    """Consecutive transition pairs from an ordered node sequence, including
    self-transitions (two consecutive attempts landing on the same node) — the
    reference directed-graph figures (Kurihara & Nakamura, LAK25) show these as
    self-loops."""
    return list(zip(nodes[:-1], nodes[1:]))


def build_aggregate_graph(
    response_df: pd.DataFrame,
    question: str,
    part_index: int = 1,
) -> tuple[list[str], Counter]:
    """Union of node labels (always including `"0"`/`"c"`) and summed transition-pair
    counts across every student's sequence for one part of `question`, aggregated over
    the whole class — the class-wide weighted directed graph."""
    question_rows = response_df[response_df["question"] == question]
    nodes: set[str] = {"0", "c"}
    edge_counts: Counter = Counter()

    for student_id in question_rows["student_id"].unique():
        seq = build_student_node_sequence(response_df, question, student_id, part_index)
        seq_nodes = seq["node"].tolist()
        nodes.update(seq_nodes)
        edge_counts.update(build_transition_pairs(seq_nodes))

    return sorted(nodes, key=node_sort_key), edge_counts


def compute_network_features(nodes: list[str], edge_counts: Counter) -> pd.DataFrame:
    """Per-node in-degree / out-degree / degree and their centrality (`degree / (n-1)`),
    computed directly from the edge-count dict — no `networkx` dependency needed since
    these graphs are always small (a handful of PRT-classification nodes)."""
    n = len(nodes)
    denom = max(n - 1, 1)

    in_degree = {node: 0 for node in nodes}
    out_degree = {node: 0 for node in nodes}
    for (src, dst), weight in edge_counts.items():
        out_degree[src] = out_degree.get(src, 0) + weight
        in_degree[dst] = in_degree.get(dst, 0) + weight

    rows = []
    for node in nodes:
        indeg = in_degree.get(node, 0)
        outdeg = out_degree.get(node, 0)
        rows.append({
            "node": node,
            "in_degree": indeg,
            "out_degree": outdeg,
            "degree": indeg + outdeg,
            "in_degree_centrality": round(indeg / denom, 4),
            "out_degree_centrality": round(outdeg / denom, 4),
            "degree_centrality": round((indeg + outdeg) / denom, 4),
        })
    return pd.DataFrame(rows)


def circular_layout(nodes: list[str]) -> dict[str, tuple[float, float]]:
    """Deterministic node positions for drawing a small transition graph: evenly spaced
    on a unit circle in `node_sort_key` order (`"0"` first, ascending numeric nodes,
    `"c"` last), independent of the input list's own order."""
    ordered = sorted(set(nodes), key=node_sort_key)
    n = len(ordered)
    if n == 0:
        return {}
    if n == 1:
        return {ordered[0]: (0.0, 0.0)}

    positions = {}
    for index, node in enumerate(ordered):
        angle = 2 * math.pi * index / n
        positions[node] = (math.cos(angle), math.sin(angle))
    return positions


def build_transition_graph_figure(
    nodes: list[str],
    edge_counts: Counter,
    colorblind_mode: bool = False,
    title: str = "",
) -> go.Figure:
    """Directed transition graph: nodes placed via `circular_layout`, edges drawn with
    width and a green (low traffic) -> red (high traffic) color scaled from transition
    count. A single student's graph (edge weights typically all 1) still renders
    correctly — width/color just collapse to the scale's midpoint."""
    positions = circular_layout(nodes)
    scale = _COLORBLIND_TRAFFIC_SCALE if colorblind_mode else _DEFAULT_TRAFFIC_SCALE

    weights = [w for w in edge_counts.values() if w > 0]
    min_w, max_w = (min(weights), max(weights)) if weights else (0, 0)

    def _style(weight: float) -> tuple[float, str]:
        norm = 0.5 if max_w == min_w else (weight - min_w) / (max_w - min_w)
        width = 1.5 + norm * 9.0
        color = px.colors.sample_colorscale(scale, [norm])[0]
        return width, color

    fig = go.Figure()
    annotations = []

    for (src, dst), weight in edge_counts.items():
        if src not in positions or dst not in positions:
            continue
        width, color = _style(weight)
        x0, y0 = positions[src]
        x1, y1 = positions[dst]

        if src == dst:
            # Self-loop: a small offset circle drawn next to the node, along the node's
            # own outward direction from the layout's center (annotation arrows can't
            # loop back on their own start point, so this is drawn as a plain line).
            norm_len = math.hypot(x0, y0) or 1.0
            ox, oy = x0 / norm_len, y0 / norm_len
            cx, cy = x0 + ox * 0.28, y0 + oy * 0.28
            loop_t = [i / 40 * 2 * math.pi for i in range(41)]
            loop_x = [cx + 0.14 * math.cos(t) for t in loop_t]
            loop_y = [cy + 0.14 * math.sin(t) for t in loop_t]
            fig.add_trace(go.Scatter(
                x=loop_x, y=loop_y, mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="text", text=f"{src} → {dst}: {weight:g}",
                showlegend=False,
            ))
            continue

        # Shrink the line endpoints toward the node radius so the arrowhead lands just
        # outside the marker instead of underneath it.
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy) or 1.0
        shrink = 0.16
        sx0, sy0 = x0 + dx / dist * shrink, y0 + dy / dist * shrink
        sx1, sy1 = x1 - dx / dist * shrink, y1 - dy / dist * shrink

        fig.add_trace(go.Scatter(
            x=[sx0, sx1], y=[sy0, sy1], mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="text", text=f"{src} → {dst}: {weight:g}",
            showlegend=False,
        ))
        annotations.append(dict(
            x=sx1, y=sy1, ax=sx0, ay=sy0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=min(width, 4),
            arrowcolor=color, standoff=6,
        ))

    ordered_nodes = sorted(set(nodes), key=node_sort_key)
    node_colors = []
    for node in ordered_nodes:
        if node == "0":
            node_colors.append("#9ca3af")
        elif node == "c":
            node_colors.append("#0072B2" if colorblind_mode else "#16a34a")
        else:
            node_colors.append("#F0E442" if colorblind_mode else "#3b82f6")

    fig.add_trace(go.Scatter(
        x=[positions[n][0] for n in ordered_nodes],
        y=[positions[n][1] for n in ordered_nodes],
        mode="markers+text",
        text=ordered_nodes,
        textposition="middle center",
        textfont=dict(color="black" if colorblind_mode else "white", size=13),
        marker=dict(size=34, color=node_colors, line=dict(width=2, color="white")),
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=title,
        annotations=annotations,
        xaxis=dict(visible=False, range=[-1.6, 1.6]),
        yaxis=dict(visible=False, range=[-1.6, 1.6], scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=420,
    )
    return fig
