from __future__ import annotations

import functools
import math
import re

import pandas as pd
import plotly.graph_objects as go

from analytics.expression_tree import parse_expression
from analytics.prt_transitions import classify_node
from analytics.tree_edit_distance import tree_edit_distance
from analytics.ui_theme import pass_fail_scale

_ANS_PATTERN = re.compile(r"ans(\d+):\s*(.*?)\s*\[(score|valid|invalid)\]")

MAX_TED_DISPLAY = 20

# Neon colorscale anchors for the 3D distance charts, in ascending distance order —
# matches Takada et al. (EDM 2025) Figures 4/5: a response sitting exactly on the correct
# answer (distance 0) is white, distance 1 is red, and it runs up through orange, yellow,
# green, and blue to black at the top of the observed range. Saturated neon hues rather
# than the plain CSS keywords, so the ramp reads as one bright turbo-style spectrum
# instead of dropping into the muddy mid-tones that `green`/`blue` render as.
# Applied per-point (not per-student), so a trajectory that starts far away and closes in
# visibly shifts color along its own length.
_DISTANCE_COLOR_ANCHORS = [
    "#FFFFFF",  # 0 — exactly correct
    "#FF1744",  # 1 — neon red
    "#FF9100",  # neon orange
    "#FFEA00",  # neon yellow
    "#39FF14",  # neon green
    "#00B0FF",  # neon blue
    "#000000",  # furthest from correct
]

# The distance-0 marker is pure white by design, which would vanish against the light
# plot panes — a thin dark outline is what keeps those "correct on the first attempt"
# dots visible, exactly as they read along the baseline of the reference figures.
_MARKER_OUTLINE = "rgba(55, 65, 81, 0.9)"

# Explicit light-blue panes (Plotly's own default 3D scene shade) rather than whatever
# Streamlit's active theme would supply — under the dark theme the panes would otherwise
# render near-black, which both loses the white distance-0 dots and stops the chart
# matching the published figures it reproduces.
_SCENE_PANE = "#E5ECF6"
_SCENE_GRID = "#FFFFFF"
_SCENE_FONT = "#2A3F5F"

# Gridlines are nudged this fraction of each axis range off their wall, toward the data,
# so they render cleanly instead of z-fighting with the coplanar wall behind them.
_GRID_INSET = 0.004

# The Students axis is drawn high-to-low so the first-ranked student sits nearest the
# viewer. Both the axis range and which face the side wall is pinned to derive from this.
_STUDENTS_AXIS_REVERSED = True

# Marks the scene's walls and gridlines, which are traces rather than layout (see
# `_static_backdrop_traces`), so anything walking a figure's traces can tell decoration
# apart from a student's trajectory. Never displayed: these traces skip hover and legend.
BACKDROP_TRACE_NAME = "__scene_backdrop__"

# Opening viewpoint for both 3D charts (Plotly's own default eye, pinned rather than
# left implicit): Students runs down to the right, Attempt down to the left, distance up.
_DEFAULT_CAMERA_EYE = dict(x=1.25, y=1.25, z=1.25)


def _raw_prt_distance(row: "pd.Series", part_index: int = 1) -> int | None:
    """Distance from the correct answer for one response to one part, or `None` if it
    belongs in the shared "other" sentinel bucket (filled in by
    `compute_prt_distance_series` once the per-question max classified distance is known).

    Derived from `classify_node` so the transition graph and the 3D chart can never
    disagree about what a response was: node label "c" is distance 0, a numeric label is
    its own distance (node 2 True -> "1" -> 1, node 3 True -> "2" -> 2, matching Takada
    et al. Table 1), and "0" (unclassified) goes to the sentinel bucket.
    """
    label = classify_node(row, part_index)
    if label == "c":
        return 0
    if label == "0":
        return None
    return int(label)


def compute_prt_distance_series(
    response_df: pd.DataFrame,
    question: str,
    part_index: int = 1,
) -> pd.DataFrame:
    """Adds a `prt_distance` column to the rows for one part of `question`, generalizing
    the teacher-authored "distance from correct answer" table in Takada et al. (EDM 2025,
    Table 1): 0 for a response with full marks on this part; `node_number - 1` for a
    response whose PRT trace terminated True at a named node; and one shared "other"
    sentinel bucket — placed 3 past the question's largest classified distance — for
    every response that terminated False, or has no PRT trace at all (blank/invalid).
    This is a heuristic reading of Table 1 (which is normally hand-calibrated per
    question by the teacher), not a teacher-calibrated distance."""
    subset = response_df[response_df["question"] == question].copy()
    if subset.empty:
        subset["prt_distance"] = pd.Series(dtype="Int64")
        return subset

    raw = subset.apply(_raw_prt_distance, axis=1, part_index=part_index)
    classified = raw.dropna()
    sentinel = (int(classified.max()) + 3) if not classified.empty else 3
    subset["prt_distance"] = raw.fillna(sentinel).astype(int)
    return subset


def _extract_expression(text: str, ans_index: int = 1) -> str | None:
    """First `ansN: <expr> [tag]` expression from a raw response/right-answer dump —
    the same pattern `analytics.latex_utils._ANS_PATTERN` matches, reused here at the
    raw-Maxima-string level instead of the LaTeX-rendering level."""
    if not text:
        return None
    for match in _ANS_PATTERN.finditer(str(text)):
        if int(match.group(1)) == ans_index:
            return match.group(2).strip()
    return None


@functools.lru_cache(maxsize=4096)
def _cached_tree_edit_distance(submitted_text: str, correct_text: str) -> int | None:
    """TED for one (submitted, correct) expression-string pair, cached since many
    students commonly submit the exact same wrong answer for a given question."""
    submitted_tree = parse_expression(submitted_text)
    correct_tree = parse_expression(correct_text)
    if submitted_tree is None or correct_tree is None:
        return None
    return tree_edit_distance(submitted_tree, correct_tree)


def compute_ted_distance_series(
    response_df: pd.DataFrame,
    question: str,
    part_index: int = 1,
) -> pd.DataFrame:
    """Adds a `ted_distance` (int or `None`) column to the rows for one part of
    `question`: the Zhang-Shasha tree edit distance between the expression the student
    submitted for that part (`ans{part_index}`) and that part's correct answer (the
    matching `ans{part_index}` in the first response row with a parseable
    `right_answer_text`). Display values are clipped at `MAX_TED_DISPLAY` (the paper:
    "answers with a TED value of 20 or higher are grouped together"). Rows whose
    submitted or correct expression can't be parsed get `ted_distance = None` and should
    be excluded from any chart built on this column."""
    subset = response_df[response_df["question"] == question].copy()
    if subset.empty:
        subset["ted_distance"] = pd.Series(dtype="Int64")
        return subset

    correct_expr_text = None
    for right_answer in subset.get("right_answer_text", pd.Series(dtype=str)):
        correct_expr_text = _extract_expression(right_answer, part_index)
        if correct_expr_text:
            break

    def _row_ted(row: "pd.Series") -> int | None:
        if not correct_expr_text:
            return None
        ans_list = row.get("ans_list") or []
        submitted_expr = next(
            (a.get("expression") for a in ans_list if a.get("index") == part_index),
            None,
        )
        if not submitted_expr:
            return None
        return _cached_tree_edit_distance(submitted_expr, correct_expr_text)

    raw = subset.apply(_row_ted, axis=1)
    subset["ted_distance"] = raw.apply(lambda v: min(v, MAX_TED_DISPLAY) if v is not None else None)
    return subset


def compute_question_student_order(distance_subset: pd.DataFrame, distance_column: str) -> dict[str, int]:
    """Ordering for the "Students" axis, matching Takada et al. (EDM 2025) Figures 4/5:
    a nested sort where each attempt breaks ties left by the previous one.

    A student's key is the full ordered tuple of their per-attempt distances, compared
    lexicographically. So the primary grouping is by the first attempt — everyone correct
    on the first try (distance 0) sits closest to the origin, then everyone who started at
    distance 1, then 2, and so on. *Within* each of those clusters the second attempt
    decides: whoever got closest on attempt two comes first, and so on recursively through
    the third, fourth, ... attempts. Python's tuple comparison gives exactly that nesting,
    and a student who simply stopped earlier sorts ahead of one who kept going from the
    same prefix (`(1, 0)` before `(1, 0, 4)`).

    This only assigns each student a position on the Students axis — it never touches the
    shape of their trajectory.
    """
    if distance_subset.empty:
        return {}

    ordered_rows = distance_subset.sort_values(by=["student_id", "completed_dt", "attempt_idx"])

    entries: list[tuple[str, tuple[float, ...]]] = []
    for student_id, group in ordered_rows.groupby("student_id"):
        # Unparseable (None/NaN) distances sort after every known value instead of being
        # treated as 0 (which would misplace them in the "correct" cluster) or crashing
        # the comparison.
        sequence = tuple(
            float(value) if pd.notna(value) else float("inf")
            for value in group[distance_column]
        )
        entries.append((student_id, sequence))

    entries.sort(key=lambda entry: (entry[1], entry[0]))
    return {student_id: rank + 1 for rank, (student_id, _sequence) in enumerate(entries)}


def _build_distance_colorscale(max_value: float) -> list[list[float | str]]:
    """Continuous Plotly colorscale anchored so that, regardless of `max_value` (the
    chart's own z-axis range), a distance of exactly 0 always renders white and exactly 1
    always renders neon red, with the remaining neon anchors (orange, yellow, green,
    blue, black) spread evenly across whatever range is left up to `max_value` — matches
    Takada et al. Figures 4/5."""
    white, red, *remaining_anchors = _DISTANCE_COLOR_ANCHORS

    if max_value <= 0:
        return [[0.0, white], [1.0, white]]

    red_position = min(1.0, 1.0 / max_value)
    stops: list[list[float | str]] = [[0.0, white], [red_position, red]]
    if red_position >= 1.0:
        # No observed distance exceeds 1 — nothing to spread the remaining anchors
        # across, and a duplicate stop position at 1.0 would fail Plotly's validation.
        return stops

    span = 1.0 - red_position
    last_index = len(remaining_anchors)
    for index, color in enumerate(remaining_anchors, start=1):
        # Force the final stop to exactly 1.0 rather than computing it — floating-point
        # accumulation in `span * index / last_index` can otherwise land a hair above
        # 1.0 (e.g. 1.0000000000000002), which Plotly's colorscale validator rejects
        # outright since stop positions must stay within the closed [0, 1] range.
        position = 1.0 if index == last_index else red_position + span * index / last_index
        stops.append([position, color])
    return stops


def _nice_step(span: float) -> int:
    """Gridline/tick spacing for the Students axis: roughly 8 divisions, snapped to a
    1/2/5-times-power-of-ten step so the labels read as round numbers."""
    if span <= 8:
        return 1
    rough = span / 8
    magnitude = 10 ** math.floor(math.log10(rough))
    for multiple in (1, 2, 5, 10):
        if rough <= multiple * magnitude:
            return int(multiple * magnitude)
    return int(10 * magnitude)


def _integer_ticks(axis_range: tuple[float, float], step: int = 1) -> list[int]:
    """Whole-number tick positions inside `axis_range`. Both the tick labels and the
    static gridlines are driven off this one list, so they can't drift apart."""
    low, high = axis_range
    start = int(math.ceil(low / step)) * step
    return list(range(start, int(math.floor(high)) + 1, step))


def _static_backdrop_traces(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    x_ticks: list[int],
    y_ticks: list[int],
    z_ticks: list[int],
) -> list[go.Scatter3d | go.Mesh3d]:
    """The three scene walls (floor, back, side) plus their gridlines, as fixed geometry.

    Plotly's built-in 3D panes are drawn on whichever faces of the cube currently face
    away from the camera, so they swap sides as soon as a drag crosses the diagonal.
    Drawing the walls as ordinary traces pins them to three fixed faces — the ones behind
    the data at the default viewpoint — so they stay where they are however the chart is
    rotated.

    Which face each wall sits on depends on the direction its axis runs: the Students axis
    is reversed (see `_build_distance_3d_figure`), which puts its far face at the *high*
    end, so the side wall goes there. Pinning it to the low end instead would leave it
    between the camera and the data, hiding half the trajectories.

    The walls are opaque. A translucent wall would let someone who orbits right around to
    the outside of the box still read the trajectories through it, but Plotly's WebGL 3D
    renderer composites translucent meshes over the entire scene without depth-sorting
    them, so a wall *behind* the data veils it too: the neon markers wash out to grey.
    Opaque walls keep the data at full saturation from every viewpoint that looks into
    the box; orbiting behind it hides the data until the modebar's reset-camera button
    (or another drag) brings the default viewpoint back.
    """
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range

    # The plane each wall lies in, and the direction that counts as "into the box" from it.
    xw, x_inward = (x1, -1.0) if _STUDENTS_AXIS_REVERSED else (x0, 1.0)
    yw, y_inward = y0, 1.0
    zw, z_inward = z0, 1.0

    quads = [
        # floor (z = zw), back wall (y = yw), side wall (x = xw)
        dict(x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[zw, zw, zw, zw]),
        dict(x=[x0, x1, x1, x0], y=[yw, yw, yw, yw], z=[z0, z0, z1, z1]),
        dict(x=[xw, xw, xw, xw], y=[y0, y1, y1, y0], z=[z0, z0, z1, z1]),
    ]
    traces: list[go.Scatter3d | go.Mesh3d] = [
        go.Mesh3d(
            **quad,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=_SCENE_PANE,
            flatshading=True,
            # Flat, unlit fill: with lighting on, the three walls catch different amounts
            # of light and read as three different shades of blue-grey.
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            hoverinfo="skip",
            showscale=False,
            showlegend=False,
            name=BACKDROP_TRACE_NAME,
        )
        for quad in quads
    ]

    # Lifted just clear of each wall, into the box, so the lines don't z-fight with the
    # coplanar mesh behind them.
    xg = xw + x_inward * _GRID_INSET * (x1 - x0)
    yg = yw + y_inward * _GRID_INSET * (y1 - y0)
    zg = zw + z_inward * _GRID_INSET * (z1 - z0)

    grid_x: list[float | None] = []
    grid_y: list[float | None] = []
    grid_z: list[float | None] = []

    def segment(start: tuple[float, float, float], end: tuple[float, float, float]) -> None:
        grid_x.extend([start[0], end[0], None])
        grid_y.extend([start[1], end[1], None])
        grid_z.extend([start[2], end[2], None])

    for x_value in x_ticks:                                   # floor + back wall
        segment((x_value, y0, zg), (x_value, y1, zg))
        segment((x_value, yg, z0), (x_value, yg, z1))
    for y_value in y_ticks:                                   # floor + side wall
        segment((x0, y_value, zg), (x1, y_value, zg))
        segment((xg, y_value, z0), (xg, y_value, z1))
    for z_value in z_ticks:                                   # back + side wall
        segment((x0, yg, z_value), (x1, yg, z_value))
        segment((xg, y0, z_value), (xg, y1, z_value))
    # Close the two open edges of the box that no gridline falls on, so the walls read as
    # a corner rather than three floating planes.
    segment((xw, yw, z0), (xw, yw, z1))

    traces.append(go.Scatter3d(
        x=grid_x, y=grid_y, z=grid_z,
        mode="lines",
        line=dict(color=_SCENE_GRID, width=2),
        hoverinfo="skip",
        showlegend=False,
        name=BACKDROP_TRACE_NAME,
    ))
    return traces


def _build_distance_3d_figure(
    distance_subset: pd.DataFrame,
    distance_column: str,
    z_title: str,
    title: str,
) -> go.Figure:
    student_order = compute_question_student_order(distance_subset, distance_column)

    plot_rows = distance_subset.dropna(subset=[distance_column]).copy()
    plot_rows = plot_rows.sort_values(by=["student_id", "completed_dt", "attempt_idx"])

    max_value = float(plot_rows[distance_column].max()) if not plot_rows.empty else 0.0
    colorscale = _build_distance_colorscale(max_value)

    fig = go.Figure()
    is_first_trace = True
    for student_id, group in plot_rows.groupby("student_id"):
        if student_id not in student_order:
            continue
        group = group.reset_index(drop=True)
        student_name = group["student_name"].iloc[0] if "student_name" in group.columns else student_id
        z_values = group[distance_column].tolist()
        fig.add_trace(go.Scatter3d(
            # Students on x, attempts on y — see `_build_distance_3d_figure`'s note on
            # which axis lands on which side of the screen.
            x=[student_order[student_id]] * len(group),
            y=list(range(len(group))),
            z=z_values,
            mode="lines+markers",
            marker=dict(
                size=5, color=z_values, colorscale=colorscale, cmin=0, cmax=max_value,
                line=dict(width=1, color=_MARKER_OUTLINE),
                showscale=is_first_trace,
                colorbar=dict(title=z_title) if is_first_trace else None,
            ),
            line=dict(width=4, color=z_values, colorscale=colorscale, cmin=0, cmax=max_value),
            showlegend=False,
            text=[str(student_name)] * len(group),
            hovertemplate="%{text}<br>Attempt %{y}<br>Rank %{x}<br>Distance %{z}<extra></extra>",
        ))
        is_first_trace = False

    max_attempts = int(plot_rows.groupby("student_id").size().max()) if not plot_rows.empty else 1
    student_count = len(student_order)
    x_range = (0.5, max(student_count, 1) + 0.5)
    y_range = (-0.5, max(max_attempts - 1, 1) + 0.5)
    z_range = (-0.5, max(max_value, 1.0) + 0.5)

    x_ticks = _integer_ticks(x_range, step=_nice_step(max(student_count, 1)))
    y_ticks = _integer_ticks(y_range)
    z_ticks = _integer_ticks(z_range)

    students_axis_range = list(reversed(x_range)) if _STUDENTS_AXIS_REVERSED else list(x_range)

    # Backdrop first, so the first *data* trace is still the one carrying the colorbar.
    for trace in _static_backdrop_traces(x_range, y_range, z_range, x_ticks, y_ticks, z_ticks):
        fig.add_trace(trace)

    axis_common = dict(
        # Plotly's own panes and gridlines are switched off: it redraws them on whichever
        # pair of cube faces currently sits behind the data, so they jump from one side of
        # the box to the other mid-drag. `_static_backdrop_traces` puts the same walls in
        # as real geometry at fixed planes instead, which stays put through any rotation.
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showspikes=False,
        color=_SCENE_FONT,
        tickmode="array",
    )
    fig.update_layout(
        title=title,
        scene=dict(
            # Students runs back-to-front: the reversed range puts rank 1 nearest the
            # viewer and rank N farthest away. This flips the coordinate space itself, so
            # every marker, line and hover label stays attached to its own student — the
            # traces are untouched, only the direction the axis is drawn in changes.
            xaxis=dict(title="Students", range=students_axis_range, tickvals=x_ticks, **axis_common),
            yaxis=dict(title="Attempt", range=list(y_range), tickvals=y_ticks, **axis_common),
            zaxis=dict(title=z_title, range=list(z_range), tickvals=z_ticks, **axis_common),
            # Fixed box proportions and a fixed opening viewpoint, so both charts open the
            # same way every render — Students running down to the right, Attempt down to
            # the left, distance up the vertical axis — and "reset camera" returns here.
            # At this eye the scene's x axis recedes to the right and y to the left, which
            # is why the student rank is plotted on x and the attempt index on y (matching
            # the axis layout of Takada et al. Figures 4 and 5), rather than the reverse.
            aspectmode="cube",
            camera=dict(eye=_DEFAULT_CAMERA_EYE, up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def build_prt_distance_3d_figure(
    response_df: pd.DataFrame,
    question: str,
    part_index: int = 1,
) -> go.Figure:
    subset = compute_prt_distance_series(response_df, question, part_index)
    return _build_distance_3d_figure(
        subset, "prt_distance", "Type of Error (PRT distance)",
        title=f"PRT-Distance Solution Process — {question} (part {part_index})",
    )


def build_ted_distance_3d_figure(
    response_df: pd.DataFrame,
    question: str,
    part_index: int = 1,
) -> go.Figure:
    subset = compute_ted_distance_series(response_df, question, part_index)
    return _build_distance_3d_figure(
        subset, "ted_distance", "Tree Edit Distance",
        title=f"TED Solution Process — {question} (part {part_index})",
    )


# ---------------------------------------------------------------------------------
# Cross-Attempt Comparison: per student, per question, how did their score/distance on
# their own retakes of this question change from their first attempt to their last?
# ---------------------------------------------------------------------------------

# Grade is stored as a 0-1 fraction (parser.py's q_score); scaled to match the 0-10
# display convention already used everywhere else in the app (see Question Analysis's
# pool_b_df["scaled_score"] = pool_b_df["grade"] * 10.0).
_GRADE_DISPLAY_SCALE = 10.0

# Which underlying per-attempt series each metric reads from, its display label/axis
# title, and whether a larger value is an improvement (Grade) or a regression (both
# distance metrics: 0 means "already correct", so smaller is better). Grade needs no
# `part_index` — parser.py's `grade` already averages every part's PRT fraction for that
# attempt — so it is the only metric that stays the same across a Select Part change.
CROSS_ATTEMPT_METRICS: dict[str, dict[str, object]] = {
    "Grade": {"axis_title": "Score (0-10)", "higher_is_better": True},
    "PRT Distance": {"axis_title": "Type of Error (PRT distance)", "higher_is_better": False},
    "Tree Edit Distance": {"axis_title": "Tree Edit Distance", "higher_is_better": False},
}

# Below this, a first-vs-last change is treated as no real change rather than a tiny
# floating-point improvement or regression — a student who scored the exact same grade
# twice shouldn't read as "Regressed" over a 1e-15 rounding artifact.
_FLAT_TOLERANCE = 1e-6

# Fixed legend/trace order, and which slot of the app's existing red/yellow/green (or
# colorblind blue/yellow/vermillion) pass/fail scale each trend maps to -- reusing
# analytics.ui_theme.pass_fail_scale exactly as the PRT pass-rate heatmap already does,
# rather than introducing a second bad/neutral/good palette into the app.
_TREND_ORDER = ["Regressed", "Flat", "Improved"]


def compute_cross_attempt_comparison(
    response_df: pd.DataFrame,
    question: str,
    metric: str,
    part_index: int = 1,
) -> pd.DataFrame:
    """Per-attempt values of `metric` (one of the keys in `CROSS_ATTEMPT_METRICS`) for
    every student on `question` who has 2+ *qualifying* attempts — one with a single
    attempt has no change to show, and is dropped rather than plotted as a lone point.

    Reuses the exact same per-attempt metrics the other Solution Process Visualization
    modules already compute (`grade` straight from the parsed response rows for the
    question overall; `compute_prt_distance_series`/`compute_ted_distance_series` for the
    two part-scoped distance metrics) instead of recomputing anything.

    Returns one row per qualifying (student, attempt) with columns `student_id`,
    `student_name`, `attempt_number` (1-based, sequential within that student's own
    attempts on this question — not a global attempt count), `value`, and `completed_dt`
    (when that attempt was submitted — lets a UI drill into one student's own attempt
    history, not just the aggregate first-vs-last comparison).
    """
    columns = ["student_id", "student_name", "attempt_number", "value", "completed_dt"]
    if metric not in CROSS_ATTEMPT_METRICS:
        raise ValueError(f"Unknown Cross-Attempt Comparison metric: {metric!r}")
    if response_df.empty:
        return pd.DataFrame(columns=columns)

    if metric == "Grade":
        subset = response_df[response_df["question"] == question].copy()
        subset["value"] = subset["grade"] * _GRADE_DISPLAY_SCALE
    elif metric == "PRT Distance":
        subset = compute_prt_distance_series(response_df, question, part_index)
        subset["value"] = subset["prt_distance"]
    else:
        subset = compute_ted_distance_series(response_df, question, part_index)
        subset["value"] = subset["ted_distance"]

    subset = subset.dropna(subset=["value"])
    if subset.empty:
        return pd.DataFrame(columns=columns)

    subset = subset.sort_values(by=["student_id", "completed_dt", "attempt_idx"])
    subset["attempt_number"] = subset.groupby("student_id").cumcount() + 1

    has_multiple_attempts = subset.groupby("student_id")["attempt_number"].transform("max") >= 2
    subset = subset[has_multiple_attempts]

    return subset[columns].reset_index(drop=True)


def classify_cross_attempt_trends(comparison: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    """One row per student: their first and last qualifying-attempt value, how many
    qualifying attempts they made (`attempt_count`), a uniformly improvement-positive
    `change` (positive = improved, negative = regressed, regardless of whether the
    underlying metric itself counts up or down when things get better — so "most
    improved" is always a plain descending sort on `change`), and a `trend` label.
    Sorted by `change` descending, i.e. most improved first.
    """
    columns = ["student_id", "student_name", "attempt_count", "first_value", "last_value", "change", "trend"]
    if comparison.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for student_id, group in comparison.groupby("student_id"):
        group = group.sort_values("attempt_number")
        first_value = float(group["value"].iloc[0])
        last_value = float(group["value"].iloc[-1])
        raw_delta = last_value - first_value
        change = raw_delta if higher_is_better else -raw_delta
        if abs(change) <= _FLAT_TOLERANCE:
            trend = "Flat"
        elif change > 0:
            trend = "Improved"
        else:
            trend = "Regressed"
        rows.append({
            "student_id": student_id,
            "student_name": group["student_name"].iloc[0],
            "attempt_count": int(group["attempt_number"].max()),
            "first_value": first_value,
            "last_value": last_value,
            "change": change,
            "trend": trend,
        })

    return pd.DataFrame(rows, columns=columns).sort_values(by="change", ascending=False).reset_index(drop=True)


def build_cross_attempt_figure(
    comparison: pd.DataFrame,
    trends: pd.DataFrame,
    metric: str,
    colorblind_mode: bool,
) -> go.Figure:
    """One line per qualifying student: x = attempt number within their own attempts on
    this question, y = the selected metric, colored by whether they improved, stayed
    flat, or regressed between their first and last attempt.

    Traces are grouped by trend (not given one legend entry per student) via
    `legendgroup`, so the legend stays exactly 3 entries — Regressed / Flat / Improved —
    however many students are plotted, and clicking one legend entry toggles every
    student in that group at once, matching the compact-legend convention the rest of
    this page's charts already follow (e.g. the transition graphs' small fixed node
    legend) rather than an unreadable one-entry-per-student list at 20-30+ students.
    """
    axis_title = CROSS_ATTEMPT_METRICS[metric]["axis_title"]
    trend_color = dict(zip(_TREND_ORDER, pass_fail_scale(colorblind_mode)))
    trend_by_student = trends.set_index("student_id")["trend"].to_dict()

    fig = go.Figure()
    for trend in _TREND_ORDER:
        student_ids = [sid for sid, t in trend_by_student.items() if t == trend]
        is_first_in_group = True
        for student_id in student_ids:
            group = comparison[comparison["student_id"] == student_id].sort_values("attempt_number")
            student_name = group["student_name"].iloc[0]
            fig.add_trace(go.Scatter(
                x=group["attempt_number"],
                y=group["value"],
                mode="lines+markers",
                line=dict(color=trend_color[trend], width=2),
                marker=dict(size=6, color=trend_color[trend]),
                opacity=0.8,
                name=trend,
                legendgroup=trend,
                showlegend=is_first_in_group,
                text=[str(student_name)] * len(group),
                hovertemplate=f"%{{text}}<br>Attempt %{{x}}<br>{axis_title}: %{{y}}<extra></extra>",
            ))
            is_first_in_group = False

    fig.update_xaxes(title="Attempt", dtick=1)
    fig.update_yaxes(title=axis_title)
    fig.update_layout(
        title=f"Cross-Attempt Comparison — {metric}",
        legend_title="Trend (first → last attempt)",
        template="plotly",
    )
    return fig


def build_single_student_attempt_figure(
    detail: pd.DataFrame,
    student_name: str,
    metric: str,
    trend: str,
    colorblind_mode: bool,
) -> go.Figure:
    """A focused view of exactly one student's own attempts on this question, for when a
    teacher clicks that student's row in the Cross-Attempt Comparison ranking table.

    `detail` is `comparison` already filtered to one `student_id`. Colored by that
    student's own trend (using the same `pass_fail_scale` slot as their line in the main
    chart), so the drill-down stays visually consistent with it and with Colorblind Mode.
    """
    axis_title = CROSS_ATTEMPT_METRICS[metric]["axis_title"]
    trend_color = dict(zip(_TREND_ORDER, pass_fail_scale(colorblind_mode)))
    color = trend_color.get(trend, trend_color["Flat"])
    ordered = detail.sort_values("attempt_number")

    fig = go.Figure(go.Scatter(
        x=ordered["attempt_number"],
        y=ordered["value"],
        mode="lines+markers",
        line=dict(color=color, width=3),
        marker=dict(size=10, color=color),
        hovertemplate=f"Attempt %{{x}}<br>{axis_title}: %{{y}}<extra></extra>",
    ))
    fig.update_xaxes(title="Attempt", dtick=1)
    fig.update_yaxes(title=axis_title)
    fig.update_layout(
        title=f"{student_name} — {metric} across attempts",
        template="plotly",
        showlegend=False,
    )
    return fig
