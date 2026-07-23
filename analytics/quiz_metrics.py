from __future__ import annotations

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


def build_scatter_figure(attempt_frame: pd.DataFrame, grade_type: str, colorblind_mode: bool = False) -> tuple[go.Figure, float, str, str] | None:
    """Attempts-vs-grade scatter, keyed by quiz_name. Returns (figure, correlation, y_label, title)."""
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
    correlation = float(merged_data["attempt_count"].corr(merged_data["overall_grade"]))

    fig = px.scatter(
        merged_data,
        x="attempt_count",
        y="overall_grade",
        color=merged_data["quiz_name"].astype(str),
        color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Set2),
        labels={"attempt_count": "No. of Attempts", "overall_grade": y_label, "color": "Quiz"},
    )
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
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
    melted = trend_data.melt("quiz_name", var_name="Metric", value_name="Value")
    fig = px.line(
        melted,
        x="quiz_name",
        y="Value",
        color="Metric",
        markers=True,
        color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Set1),
        labels={"quiz_name": "Quiz", "Value": "Value"},
    )
    fig.update_xaxes(type="category")
    fig.update_layout(title="Line Graph of Various Metrics")
    return fig
