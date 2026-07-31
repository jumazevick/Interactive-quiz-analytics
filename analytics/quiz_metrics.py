from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.dates import date2num, num2date
from scipy.stats import gaussian_kde

from analytics.ui_theme import qualitative_colors

# Okabe-Ito vermillion — a colorblind-safer stand-in for the default reddish mean-grade
# overlay line/marker color.
_COLORBLIND_ACCENT = "#D55E00"
_DEFAULT_ACCENT = "#FF474C"

ATTEMPT_FRAME_COLUMNS = ["quiz_name", "student_name", "student_id", "attempt_idx", "overall_grade", "completed_dt", "started_on"]


_LABEL_TOKEN_RE = re.compile(r"[^\s_-]+[\s_-]*")


def _wrap_category_label(label: str, max_chars: int = 22, max_lines: int = 2) -> str:
    """Wrap a long quiz name onto up to `max_lines` short horizontal lines (joined
    with `<br>`) instead of leaving Plotly to render it as one long diagonal tick
    label that eats most of the chart's vertical space. Anything left over past
    `max_lines` is elided with an ellipsis — the full name is still available on
    hover, this only affects the printed tick label.

    Tokenizes on underscores and hyphens as well as whitespace: these STACK quiz
    names are commonly one single space-free token like
    `mat1_numeri_complessi-Espressioni`, all sharing the same `mat1_numeri_complessi-`
    prefix — wrapping on whitespace alone would treat the whole name as one
    unbreakable word and truncate it down to just that shared, non-distinguishing
    prefix, hiding the one part (`Espressioni`) that actually differs between quizzes.
    """
    tokens = _LABEL_TOKEN_RE.findall(str(label))
    if not tokens:
        return str(label)
    lines: list[str] = []
    current = ""
    remaining = list(tokens)
    while remaining and len(lines) < max_lines:
        token = remaining[0]
        if not current and len(token) > max_chars:
            lines.append(token[: max_chars - 1].rstrip() + "…")
            remaining.pop(0)
            continue
        candidate = current + token
        if len(candidate) > max_chars and current:
            lines.append(current.rstrip())
            current = ""
            continue
        current = candidate
        remaining.pop(0)
    if current and len(lines) < max_lines:
        lines.append(current.rstrip())
    if remaining and not (lines and lines[-1].endswith("…")):
        if lines:
            lines[-1] = lines[-1].rstrip() + "…"
        else:
            lines = [current[:max_chars].rstrip() + "…"]
    return "<br>".join(lines)


def build_quiz_attempt_frame(response_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long per-question response_df to one row per attempt, across all
    uploaded quiz files. Dedupes on (quiz_name, attempt_idx) rather than attempt_idx
    alone, since attempt_idx is only assigned uniquely within a single uploaded file."""
    if response_df.empty:
        return pd.DataFrame(columns=ATTEMPT_FRAME_COLUMNS)
    return (
        response_df[ATTEMPT_FRAME_COLUMNS]
        .drop_duplicates(subset=["quiz_name", "attempt_idx"])
        .reset_index(drop=True)
    )


def compute_quiz_stats(attempt_frame: pd.DataFrame, selected_stats: list[str]) -> pd.DataFrame:
    """Same formulas as the original per-file Quiz Analysis page, grouped by quiz_name
    instead of quizID and reading overall_grade instead of a locally normalized grade."""
    if attempt_frame.empty:
        return pd.DataFrame()

    dfs = []

    if "student_count" in selected_stats:
        students_per_quiz = attempt_frame.groupby("quiz_name")["student_id"].nunique().reset_index()
        students_per_quiz.columns = ["quiz_name", "student_count"]
        dfs.append(students_per_quiz)

    if "mean_grade" in selected_stats or "grade_variance" in selected_stats:
        grade_statistics = attempt_frame.groupby("quiz_name")["overall_grade"].agg(["mean", "var"]).reset_index()
        grade_statistics.columns = ["quiz_name", "mean_grade", "grade_variance"]
        if "mean_grade" not in selected_stats:
            grade_statistics = grade_statistics.drop(columns=["mean_grade"])
        if "grade_variance" not in selected_stats:
            grade_statistics = grade_statistics.drop(columns=["grade_variance"])
        dfs.append(grade_statistics)

    if "mean_highest_grade" in selected_stats:
        highest_grades = attempt_frame.groupby(["quiz_name", "student_id"])["overall_grade"].max().reset_index()
        average_highest_grades = highest_grades.groupby("quiz_name")["overall_grade"].mean().reset_index()
        average_highest_grades.columns = ["quiz_name", "mean_highest_grade"]
        dfs.append(average_highest_grades)

    if "attempt_count" in selected_stats:
        total_attempts_per_quiz = attempt_frame.groupby("quiz_name").size().reset_index(name="attempt_count")
        dfs.append(total_attempts_per_quiz)

    if "attempt_rate" in selected_stats:
        attempts_per_student = attempt_frame.groupby(["quiz_name", "student_id"]).size().reset_index(name="attempt_count")
        average_attempts_per_student = attempts_per_student.groupby("quiz_name")["attempt_count"].mean().reset_index()
        average_attempts_per_student.columns = ["quiz_name", "attempt_rate"]
        dfs.append(average_attempts_per_student)

    if dfs:
        quiz_stats = dfs[0]
        for frame in dfs[1:]:
            quiz_stats = pd.merge(quiz_stats, frame, on="quiz_name")
        return quiz_stats.round(2)
    return pd.DataFrame()


def build_boxplot_figure(attempt_frame: pd.DataFrame, colorblind_mode: bool = False) -> go.Figure:
    """Grade distribution per quiz, with an overlaid mean_grade line — same construction
    as the original Quiz Analysis boxplot, keyed by quiz_name.

    Without an explicit `color=`, px.box puts every box in a single trace sharing one
    color (just the first palette entry) regardless of `color_discrete_sequence` — every
    quiz rendered identically. `color="quiz_name"` gives each quiz its own trace/color;
    the previous hardcoded black point markers were also invisible against a dark theme.
    """
    fig = px.box(
        attempt_frame,
        x="quiz_name",
        y="overall_grade",
        points="all",
        color="quiz_name",
        color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Bold),
        labels={"quiz_name": "Quiz", "overall_grade": "Grade"},
    )
    fig.update_traces(marker=dict(size=4, opacity=0.6), jitter=0.3)

    accent = _COLORBLIND_ACCENT if colorblind_mode else _DEFAULT_ACCENT
    means = attempt_frame.groupby("quiz_name")["overall_grade"].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=means["quiz_name"],
        y=means["overall_grade"],
        mode="lines+markers",
        name="mean_grade",
        line=dict(color=accent, width=2),
        marker=dict(size=8, color=accent),
    ))
    fig.update_layout(title="Grade Distribution")
    # Full quiz names as x-axis tick labels force Plotly to render them diagonally,
    # which can eat most of the chart's vertical space and squeeze the actual plot
    # into a tiny strip — the per-quiz legend (color="quiz_name" above) already
    # identifies each box, so the tick labels are redundant; hidden from display but
    # still available on hover.
    fig.update_xaxes(showticklabels=False, title=None)
    return fig


def build_engagement_figure(attempt_frame: pd.DataFrame, colorblind_mode: bool = False) -> go.Figure | None:
    """Per-quiz gaussian KDE (Scott's rule bandwidth) of attempt start dates — same
    method seaborn.kdeplot uses internally. Returns None if there's nothing plottable."""
    if attempt_frame.empty or attempt_frame["started_on"].isna().any():
        return None

    fig = go.Figure(layout=dict(colorway=qualitative_colors(colorblind_mode, px.colors.qualitative.Plotly)))
    for quiz_name in attempt_frame["quiz_name"].unique():
        quiz_data = attempt_frame[attempt_frame["quiz_name"] == quiz_name]
        if quiz_data.empty:
            continue
        dates_numeric = date2num(quiz_data["started_on"])
        try:
            kde = gaussian_kde(dates_numeric)
        except np.linalg.LinAlgError:
            continue
        grid = np.linspace(dates_numeric.min(), dates_numeric.max(), 200)
        density = kde(grid)
        fig.add_trace(go.Scatter(x=num2date(grid), y=density, mode="lines", name=str(quiz_name), fill="tozeroy"))

    if not fig.data:
        return None

    fig.update_layout(title="Engagement Over Time", xaxis_title="Date", yaxis_title="Frequency Density")
    return fig


# Marker size range for the Attempts vs Grades scatter's density cue (see
# `build_scatter_figure`): a point on its own renders at the low end, a coordinate
# shared by several students renders larger, clamped so the largest overlaps don't
# balloon into a dominant bubble chart.
_SCATTER_MARKER_SIZE_MIN = 12
_SCATTER_MARKER_SIZE_MAX = 22
_SCATTER_MARKER_SIZE_SATURATES_AT = 6  # student count at which size stops growing


def _deterministic_jitter(keys: pd.Series, amplitude: float, salt: str) -> pd.Series:
    """A small, stable pseudo-random offset per key, in the range [-amplitude, amplitude].

    Seeded from each row's own key via a content hash rather than numpy's global RNG
    state, so the same student gets the same offset on every rerun -- jitter that
    reshuffled on each interaction would be worse than the overlap it's meant to fix.
    `salt` decorrelates the offset used for the x-axis from the one used for the y-axis:
    reusing the same offset for both would just slide an overlapping stack diagonally
    as a block instead of fanning it out.
    """
    def offset(key: str) -> float:
        digest = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
        fraction = int(digest[:8], 16) / 0xFFFFFFFF  # stable float in [0, 1]
        return (fraction * 2 - 1) * amplitude

    return keys.astype(str).map(offset)


def build_scatter_figure(attempt_frame: pd.DataFrame, grade_type: str, colorblind_mode: bool = False) -> tuple[go.Figure, float, str, str] | None:
    """Attempts-vs-grade scatter, keyed by quiz_name. Returns (figure, correlation, y_label, title).

    Many students share an exact (attempts, grade) coordinate -- most obviously at low
    attempt counts, where "passed on the first try" is common -- so without help, points
    from different quizzes stack exactly on top of each other and only the topmost color
    is visible. Three things address that together, matching what the true coordinates
    would otherwise hide: a small deterministic jitter so an overlapping stack fans out
    into a visible cluster, reduced opacity so any jitter can't fully resolve still reads
    as a denser/darker region, and marker size scaling mildly with how many students
    shared that exact coordinate before jitter was applied.
    """
    if attempt_frame.empty:
        return None

    attempt_count = attempt_frame.groupby(["quiz_name", "student_id"]).size().reset_index(name="attempt_count")

    if grade_type == "Highest Grade":
        grade_data = attempt_frame.groupby(["quiz_name", "student_id"])["overall_grade"].max().reset_index()
        y_label, title = "Highest Grade", "Attempts vs Highest Grade"
    elif grade_type == "Minimum Grade":
        grade_data = attempt_frame.groupby(["quiz_name", "student_id"])["overall_grade"].min().reset_index()
        y_label, title = "Minimum Grade", "Attempts vs Minimum Grade"
    else:
        grade_data = attempt_frame.groupby(["quiz_name", "student_id"])["overall_grade"].mean().reset_index()
        y_label, title = "Average Grade", "Attempts vs Average Grade"

    merged_data = pd.merge(attempt_count, grade_data, on=["quiz_name", "student_id"])
    # Correlation is computed from the true, unjittered values before anything below
    # touches the DataFrame — the reported r must describe the real data, not the
    # display-only fan-out applied to the plotted points.
    correlation = float(merged_data["attempt_count"].corr(merged_data["overall_grade"]))

    merged_data["quiz_name_str"] = merged_data["quiz_name"].astype(str)
    jitter_key = merged_data["quiz_name_str"] + "|" + merged_data["student_id"].astype(str)
    # +/-0.15 keeps the true integer attempt count still visually obvious (the task's own
    # target) while being enough to separate an exact-overlap stack into a visible cluster.
    merged_data["attempt_count_plot"] = merged_data["attempt_count"] + _deterministic_jitter(jitter_key, amplitude=0.15, salt="x")
    merged_data["overall_grade_plot"] = merged_data["overall_grade"] + _deterministic_jitter(jitter_key, amplitude=0.15, salt="y")

    group_sizes = merged_data.groupby(["quiz_name_str", "attempt_count", "overall_grade"])["student_id"].transform("size")
    saturation = (group_sizes.clip(upper=_SCATTER_MARKER_SIZE_SATURATES_AT) - 1) / (_SCATTER_MARKER_SIZE_SATURATES_AT - 1)
    merged_data["_marker_size"] = _SCATTER_MARKER_SIZE_MIN + saturation * (_SCATTER_MARKER_SIZE_MAX - _SCATTER_MARKER_SIZE_MIN)

    fig = px.scatter(
        merged_data,
        x="attempt_count_plot",
        y="overall_grade_plot",
        color="quiz_name_str",
        size="_marker_size",
        size_max=_SCATTER_MARKER_SIZE_MAX,
        color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Set2),
        labels={"attempt_count_plot": "No. of Attempts", "overall_grade_plot": y_label, "quiz_name_str": "Quiz"},
        custom_data=["attempt_count", "overall_grade"],
    )
    # Hover shows the true (pre-jitter) coordinates — a student's actual attempt count
    # and grade — never the display-only jittered position.
    fig.update_traces(
        marker=dict(opacity=0.65, line=dict(width=1, color="white")),
        hovertemplate="Attempts: %{customdata[0]}<br>" + y_label + ": %{customdata[1]}<extra></extra>",
    )
    # Ticks at true integer attempt counts, not the jittered fractional positions the
    # points themselves sit at.
    fig.update_xaxes(tickmode="linear", dtick=1)
    fig.update_layout(title=title, legend_title="Quiz")
    return fig, correlation, y_label, title


def build_metric_trend_data(attempt_frame: pd.DataFrame, selected_metrics: list[str]) -> pd.DataFrame:
    data = {}
    if "student_count" in selected_metrics:
        data["student_count"] = attempt_frame.groupby("quiz_name")["student_id"].nunique()
    if "attempt_rate" in selected_metrics:
        attempts_per_student = attempt_frame.groupby(["quiz_name", "student_id"]).size().reset_index(name="attempt_count")
        data["attempt_rate"] = attempts_per_student.groupby("quiz_name")["attempt_count"].mean()
    if "mean_grade" in selected_metrics:
        data["mean_grade"] = attempt_frame.groupby("quiz_name")["overall_grade"].mean()
    if "grade_variance" in selected_metrics:
        data["grade_variance"] = attempt_frame.groupby("quiz_name")["overall_grade"].var()
    return pd.DataFrame(data).reset_index()


def build_line_graph_figure(trend_data: pd.DataFrame, colorblind_mode: bool = False) -> go.Figure:
    # Unlike the boxplot (color="quiz_name"), this chart colors by Metric, so the
    # quiz-name x-axis labels are the only way to tell which quiz a point belongs to
    # — hiding them outright (as for the boxplot) isn't an option. Wrapping them onto
    # a couple of short horizontal lines instead of one long diagonal string keeps
    # them readable without the label eating most of the chart's vertical space.
    melted = trend_data.melt("quiz_name", var_name="Metric", value_name="Value")
    melted["quiz_name"] = melted["quiz_name"].map(_wrap_category_label)
    fig = px.line(
        melted,
        x="quiz_name",
        y="Value",
        color="Metric",
        markers=True,
        color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Set1),
        labels={"quiz_name": "Quiz", "Value": "Value"},
    )
    fig.update_xaxes(type="category", tickangle=0, tickfont=dict(size=10))
    # Give each category more horizontal room as quiz count grows, so wrapped labels
    # (which share a long common prefix, e.g. "mat1_numeri_complessi-") don't crowd
    # into their neighbors — Streamlit's use_container_width on-screen ignores this
    # in favor of the container's actual width, but the PDF export (a fixed-size
    # rasterization) respects it directly.
    n_categories = max(trend_data["quiz_name"].nunique(), 1)
    fig.update_layout(title="Line Graph of Various Metrics", width=max(800, 220 * n_categories))
    return fig
