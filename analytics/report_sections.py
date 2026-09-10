from __future__ import annotations

import re

import pandas as pd
import plotly.express as px

from analytics.latex_utils import extract_stack_answer_latex
from analytics.prt_transitions import (
    build_aggregate_graph,
    build_transition_graph_figure,
    compute_network_features,
    count_question_parts,
)
from analytics.parser import get_attempt_pools
from analytics.prt_analysis import build_prt_pass_heatmap, build_prt_pass_heatmap_figure
from analytics.question_analytics import build_question_analytics
from analytics.question_details import build_error_drilldown
from analytics.quiz_metrics import (
    build_boxplot_figure,
    build_engagement_figure,
    build_line_graph_figure,
    build_metric_trend_data,
    build_quiz_attempt_frame,
    build_scatter_figure,
    compute_quiz_stats,
)
from analytics.solution_distance import (
    CROSS_ATTEMPT_METRICS,
    build_cross_attempt_figure,
    build_prt_distance_3d_figure,
    build_ted_distance_3d_figure,
    classify_cross_attempt_trends,
    compute_cross_attempt_comparison,
)
from analytics.ui_theme import humanize_columns, qualitative_colors

# The canonical module list for each of the three analysis sections, in report order.
# One list per section, used both for the PDF panel's multiselects and to gate the
# builders below, so the two can never drift apart.
QUESTION_MODULES = [
    "1. Question Summary",
    "2. Question Difficulty Analysis",
    "3. Question Item Details & Error Drill-Down",
    "4. Question Response Distribution",
    "5. Student Performance Matrix",
    "6. Question Metrics",
]

SPV_MODULES = [
    "Class-Wide Transition Graph",
    "Network Features per Node",
    "PRT-Distance 3D Chart",
    "Tree Edit Distance 3D Chart",
    "Cross-Attempt Comparison",
]

# The report has no page-side control for which metric to compare by (unlike the SPV
# page's sidebar radio), so it uses the same metric the on-screen module opens with —
# Grade, the default there, which also has the advantage of not depending on which part
# is selected.
_DEFAULT_CROSS_ATTEMPT_METRIC = "Grade"

QUIZ_MODULES = [
    "1. Merged List of Users and Files",
    "2. Summary of Quiz Stats",
    "3. Quiz Grade Distribution (Box Plot)",
    "4. Engagement Over Time",
    "5. Scatter Plot: Attempts vs Grades",
    "6. Line Graph of Various Metrics",
]

# The Quiz Analysis sidebar lets the reader narrow these three on-screen; the report
# deliberately always uses the full defaults instead of whatever that sidebar happens to
# hold, so the same selections produce the same PDF from any page.
_DEFAULT_QUIZ_STATS = [
    "student_count", "attempt_rate", "mean_grade",
    "grade_variance", "mean_highest_grade", "attempt_count",
]
_DEFAULT_QUIZ_METRICS = ["student_count", "attempt_rate", "mean_grade", "grade_variance"]
_DEFAULT_GRADE_TYPE = "Average Grade"


def _q_num(q_name: str) -> int:
    m = re.search(r"\d+", str(q_name))
    return int(m.group(0)) if m else 0


def build_question_pdf_sections(
    response_df: pd.DataFrame,
    quiz_name: str,
    selected_sections: list[str],
    colorblind_mode: bool,
) -> list[dict]:
    """All 6 Question Analysis PDF sections (summary, difficulty w/ charts, item
    details/error drill-down, response distribution, student matrix, metrics) for
    any uploaded quiz, gated by `selected_sections` — used for every quiz in the
    PDF's breakdown (including the one currently shown on-screen) so each quiz
    gets an identical, complete 1-6 section run before the next quiz starts."""
    quiz_df = response_df[response_df["quiz_name"] == quiz_name].copy()
    if quiz_df.empty:
        return []

    quiz_analytics = build_question_analytics(quiz_df, str(quiz_name))
    q_metrics = quiz_analytics["question_metrics"]
    q_difficulty = quiz_analytics["difficulty_metrics"]
    q_response_outcomes = quiz_analytics["response_outcomes"]
    q_repeated_wrong = quiz_analytics["repeated_wrong_answers"]
    q_prt_pass_rates = quiz_analytics["prt_pass_rates"]
    q_pool_b = quiz_analytics["pool_b_df"].copy()
    q_ranked_difficulty = quiz_analytics["ranked_difficulty"]

    if q_pool_b.empty:
        return []

    q_pool_b["scaled_score"] = q_pool_b["grade"] * 10.0
    q_order = sorted(q_pool_b["question"].unique(), key=_q_num)
    prefix = f"{quiz_name} — "

    sections: list[dict] = []

    if "1. Question Summary" in selected_sections:
        sections.append({
            "title": f"{prefix}1. Question Summary",
            "caption": "Participation and summary statistics",
            "df": humanize_columns(q_metrics[["question", "attempts", "students", "percent_valid", "percent_invalid", "syntax_error_count"]]),
        })

    if "2. Question Difficulty Analysis" in selected_sections:
        difficulty_charts = []
        fig = px.bar(
            q_ranked_difficulty.head(10), x="question", y="avg_score", color="question",
            color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Set2),
            labels={"avg_score": "Average score", "question": "Question"},
        )
        fig.update_layout(title="Top Difficult Questions by Average Score", showlegend=False, template="plotly")
        difficulty_charts.append({"title": "Top Difficult Questions by Average Score", "figure": fig})

        fig2 = px.box(
            q_pool_b, x="question", y="scaled_score", color="question",
            color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Set2),
            labels={"scaled_score": "Score (0-10)", "question": "Question"},
        )
        fig2.update_layout(title="Score Distribution by Question (Best Attempt per Student)", showlegend=False, template="plotly")
        difficulty_charts.append({"title": "Score Distribution by Question (Best Attempt per Student)", "figure": fig2})

        sections.append({
            "title": f"{prefix}2. Question Difficulty Analysis",
            "caption": "Facility and discrimination (Best Attempt)",
            "df": humanize_columns(q_difficulty),
            "charts": difficulty_charts,
        })

    if "3. Question Item Details & Error Drill-Down" in selected_sections:
        item_details_rows = []
        for q in q_order:
            drilldown = build_error_drilldown(q_pool_b, q)
            if not drilldown.empty:
                flat = drilldown.copy()
                flat.insert(0, "Question", q)
                flat["Submitted Response"] = flat["Submitted Response"].apply(extract_stack_answer_latex)
                flat["Right Answer"] = flat["Right Answer"].apply(extract_stack_answer_latex)
                item_details_rows.append(flat)
        item_details_table = pd.concat(item_details_rows, ignore_index=True) if item_details_rows else pd.DataFrame()
        sections.append({
            "title": f"{prefix}3. Question Item Details & Error Drill-Down",
            "caption": "Question text, right answer, and wrong-response drill-down (Best Attempt)",
            "df": humanize_columns(item_details_table),
        })

    if "4. Question Response Distribution" in selected_sections:
        response_charts = []
        has_prt_data = bool(any(str(row.get("response_text", "")).strip() for _, row in quiz_df.iterrows()))
        if has_prt_data:
            valid_invalid_q = pd.DataFrame({
                "question": q_metrics["question"],
                "Valid %": q_metrics["percent_valid"],
                "Invalid/Syntax Error %": q_metrics["percent_invalid"],
            })
            fig = px.bar(
                q_response_outcomes, x="question", y=["correct_percent", "incorrect_percent"],
                barmode="group", color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Vivid),
                labels={"value": "Percent", "question": "Question"},
            )
            fig.update_layout(title="Response Outcome Percentages (Best Attempts)", template="plotly")
            response_charts.append({"title": "Response Outcome Percentages (Best Attempts)", "figure": fig})

            fig2 = px.bar(
                valid_invalid_q, x="question", y=["Valid %", "Invalid/Syntax Error %"],
                barmode="group", color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Vivid),
                labels={"value": "Percent", "question": "Question"},
            )
            fig2.update_layout(title="Valid vs Invalid Attempts (All Attempts)", template="plotly")
            response_charts.append({"title": "Valid vs Invalid Attempts (All Attempts)", "figure": fig2})

            heatmap_df = build_prt_pass_heatmap(q_prt_pass_rates, q_order, quiz_analytics["prt_frame"])
            if not heatmap_df.empty and len(heatmap_df.columns):
                fig3 = build_prt_pass_heatmap_figure(heatmap_df, colorblind_mode)
                response_charts.append({"title": "PRT Pass Heatmap", "figure": fig3})

        sections.append({
            "title": f"{prefix}4. Question Response Distribution",
            "caption": "Response outcomes and top wrong answers",
            "df": humanize_columns(q_response_outcomes.merge(q_repeated_wrong.drop(columns=["top_wrong_expressions"], errors="ignore"), on="question", how="left")),
            "charts": response_charts,
        })

    if "5. Student Performance Matrix" in selected_sections:
        student_matrix_q = q_pool_b.pivot_table(
            index="student_id", columns="question", values="grade",
            aggfunc="first", fill_value=0.0, dropna=False,
        ).reindex(columns=q_order, fill_value=0.0)

        fig = px.imshow(student_matrix_q, labels=dict(x="Question", y="Student", color="Score"), color_continuous_scale="Viridis")
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(student_matrix_q.columns))), ticktext=[str(c) for c in student_matrix_q.columns])
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(student_matrix_q.index))), ticktext=[str(r) for r in student_matrix_q.index])
        chart_height = max(400, 24 * len(student_matrix_q.index))
        fig.update_layout(title="Student-by-Question Performance Matrix (Best Attempts)", height=chart_height, template="plotly")

        sections.append({
            "title": f"{prefix}5. Student Performance Matrix",
            "caption": "Per-student score per question (Best Attempt)",
            "df": humanize_columns(student_matrix_q.reset_index()),
            "charts": [{"title": "Student-by-Question Performance Matrix (Best Attempts)", "figure": fig}],
        })

    if "6. Question Metrics" in selected_sections:
        q_metrics_flat = q_metrics.merge(
            q_difficulty[["question", "discrimination_index", "average_marks", "median_marks", "standard_deviation"]],
            on="question", how="left",
        )
        q_metrics_export = q_metrics_flat[[
            "question", "attempts", "students", "invalid_rate", "blank_rate",
            "reattempt_share", "facility", "partial_credit_mean",
            "discrimination_index", "average_marks", "median_marks", "standard_deviation",
            "catch_all_share",
        ]].rename(columns={"discrimination_index": "discrimination"})
        sections.append({
            "title": f"{prefix}6. Question Metrics",
            "caption": "Consolidated question analytics table",
            "df": humanize_columns(q_metrics_export),
        })

    return sections


def build_spv_pdf_sections(
    response_df: pd.DataFrame,
    quiz_name: str,
    question: str,
    part_index: int,
    selected_sections: list[str],
    colorblind_mode: bool,
) -> list[dict]:
    """The Solution Process Visualization sections for one quiz, question and part.

    Builds its own figures rather than reusing whatever the page happened to render, so the
    report is identical from any page — including pages where these charts were never drawn.
    The single-student transition graph is deliberately absent: it is an interactive,
    click-a-row view with no meaningful default.
    """
    quiz_df = response_df[response_df["quiz_name"] == quiz_name].copy()
    if quiz_df.empty:
        return []

    pool_a_df, _ = get_attempt_pools(quiz_df)
    if pool_a_df.empty or question not in set(pool_a_df["question"]):
        return []

    # A part the caller asked for may not exist on this quiz's question; fall back to the
    # last one that does rather than returning an empty section.
    part_index = min(part_index, count_question_parts(pool_a_df, question))
    prefix = f"{quiz_name} — {question} part {part_index} — "

    sections: list[dict] = []
    agg_nodes, agg_edges = build_aggregate_graph(pool_a_df, question, part_index)

    if "Class-Wide Transition Graph" in selected_sections and agg_edges:
        fig = build_transition_graph_figure(
            agg_nodes, agg_edges, colorblind_mode=colorblind_mode,
            title=f"Class-wide Answer Transitions — {question} (part {part_index})",
        )
        sections.append({
            "title": f"{prefix}Class-Wide Transition Graph",
            "caption": "Aggregated solution-process transition graph; edge thickness and color scale with how many students made each transition",
            "charts": [{"title": "Transition Graph", "figure": fig}],
        })

    if "Network Features per Node" in selected_sections and agg_edges:
        network_features = compute_network_features(agg_nodes, agg_edges)
        node_order = network_features["node"].tolist()
        feature_charts = []
        for metric, label in (
            ("in_degree_centrality", "In-Degree Centrality"),
            ("out_degree_centrality", "Out-Degree Centrality"),
            ("degree_centrality", "Degree Centrality"),
        ):
            feature_fig = px.bar(
                network_features, x="node", y=metric,
                category_orders={"node": node_order},
                labels={"node": "Node", metric: label},
            )
            feature_fig.update_traces(marker_color="#3b82f6")
            feature_fig.update_xaxes(type="category")
            feature_fig.update_layout(title=label, showlegend=False)
            feature_charts.append({"title": label, "figure": feature_fig})
        sections.append({
            "title": f"{prefix}Network Features",
            "caption": "In-degree / out-degree / degree centrality per node",
            "df": humanize_columns(network_features),
            "charts": feature_charts,
        })

    distance_charts = []
    if "PRT-Distance 3D Chart" in selected_sections:
        distance_charts.append({
            "title": "PRT Distance",
            "figure": build_prt_distance_3d_figure(pool_a_df, question, part_index),
        })
    if "Tree Edit Distance 3D Chart" in selected_sections:
        distance_charts.append({
            "title": "Tree Edit Distance",
            "figure": build_ted_distance_3d_figure(pool_a_df, question, part_index),
        })
    if distance_charts:
        sections.append({
            "title": f"{prefix}3D Solution Process Distance",
            "caption": "One trajectory per student over Attempt x Students x distance from the correct answer, each point colored by its own distance",
            "charts": distance_charts,
        })

    if "Cross-Attempt Comparison" in selected_sections:
        metric = _DEFAULT_CROSS_ATTEMPT_METRIC
        higher_is_better = bool(CROSS_ATTEMPT_METRICS[metric]["higher_is_better"])
        comparison = compute_cross_attempt_comparison(pool_a_df, question, metric, part_index)
        if not comparison.empty:
            trends = classify_cross_attempt_trends(comparison, higher_is_better)
            counts = trends["trend"].value_counts()
            fig = build_cross_attempt_figure(comparison, trends, metric, colorblind_mode)
            ranking_table = trends.rename(columns={
                "student_name": "Student Name",
                "attempt_count": "Attempts",
                "first_value": "First Attempt",
                "last_value": "Last Attempt",
                "change": "Change",
                "trend": "Trend",
            })[["Student Name", "Attempts", "First Attempt", "Last Attempt", "Change", "Trend"]]
            sections.append({
                "title": f"{prefix}Cross-Attempt Comparison ({metric})",
                "caption": (
                    f"Per-student {metric.lower()} change from first to last attempt among students "
                    f"with 2+ attempts: {int(counts.get('Improved', 0))} improved, "
                    f"{int(counts.get('Flat', 0))} flat, {int(counts.get('Regressed', 0))} regressed"
                ),
                "df": humanize_columns(ranking_table),
                "charts": [{"title": "Cross-Attempt Comparison", "figure": fig}],
            })

    return sections


def build_quiz_pdf_sections(
    response_df: pd.DataFrame,
    quiz_names: list[str],
    selected_sections: list[str],
    colorblind_mode: bool,
) -> list[dict]:
    """The Quiz Analysis sections, combined across `quiz_names`.

    Same figure-building code the page renders on screen, moved here so the report does not
    depend on those charts having been drawn — on the Question Analysis page they never are.
    """
    attempt_frame = build_quiz_attempt_frame(response_df[response_df["quiz_name"].isin(quiz_names)])
    if attempt_frame.empty:
        return []

    sections: list[dict] = []

    if "1. Merged List of Users and Files" in selected_sections:
        sections.append({
            "title": "1. Merged List of Users and Files",
            "caption": "All parsed quiz attempt rows (combined across uploaded files)",
            "df": humanize_columns(attempt_frame),
        })

    if "2. Summary of Quiz Stats" in selected_sections:
        sections.append({
            "title": "2. Summary of Quiz Stats",
            "caption": "Aggregated stats per quiz",
            "df": humanize_columns(compute_quiz_stats(attempt_frame, _DEFAULT_QUIZ_STATS)),
        })

    if "3. Quiz Grade Distribution (Box Plot)" in selected_sections:
        fig = build_boxplot_figure(attempt_frame, colorblind_mode=colorblind_mode)
        fig.update_layout(template="plotly")
        sections.append({
            "title": "3. Quiz Grade Distribution (Box Plot)",
            "caption": "Spread of grades per quiz, with mean grade overlay",
            "charts": [{"title": "Grade Distribution", "figure": fig}],
        })

    if "4. Engagement Over Time" in selected_sections:
        fig = build_engagement_figure(attempt_frame, colorblind_mode=colorblind_mode)
        if fig is not None:
            fig.update_layout(template="plotly")
            sections.append({
                "title": "4. Engagement Over Time",
                "caption": "Density of quiz attempt start times per quiz",
                "charts": [{"title": "Engagement Over Time", "figure": fig}],
            })

    if "5. Scatter Plot: Attempts vs Grades" in selected_sections:
        result = build_scatter_figure(attempt_frame, _DEFAULT_GRADE_TYPE, colorblind_mode=colorblind_mode)
        if result is not None:
            fig, correlation, y_label, _ = result
            fig.update_layout(template="plotly")
            sections.append({
                "title": "5. Scatter Plot: Attempts vs Grades",
                "caption": f"Correlation between number of attempts and quiz {y_label}: r = {correlation:.2f}",
                "charts": [{"title": "Attempts vs Grades", "figure": fig}],
            })

    if "6. Line Graph of Various Metrics" in selected_sections:
        trend_data = build_metric_trend_data(attempt_frame, _DEFAULT_QUIZ_METRICS)
        fig = build_line_graph_figure(trend_data, colorblind_mode=colorblind_mode)
        fig.update_layout(template="plotly")
        sections.append({
            "title": "6. Line Graph of Various Metrics",
            "caption": "Trend of selected metrics across quizzes",
            "charts": [{"title": "Metrics by Quiz", "figure": fig}],
        })

    return sections


def build_report_sections(
    response_df: pd.DataFrame,
    selected_quizzes: list[str],
    colorblind_mode: bool,
    question_sections: list[str] | None,
    spv_sections: list[str] | None,
    quiz_sections: list[str] | None,
    spv_question: str | None = None,
    spv_part: int = 1,
) -> list[dict]:
    """Assemble the whole report, in the documented order:

    1. Question Analysis — for each selected quiz in turn, that quiz's complete run of
       selected modules, before the next quiz starts.
    2. Solution Process Visualization — the selected modules, per selected quiz.
    3. Quiz Analysis — the selected modules, aggregated across the selected quizzes.

    A `None` section list means that top-level part is excluded entirely; an empty list
    means it was included with nothing ticked, which also yields nothing.
    """
    sections: list[dict] = []

    if question_sections:
        for quiz_name in selected_quizzes:
            sections.extend(build_question_pdf_sections(
                response_df, quiz_name, question_sections, colorblind_mode,
            ))

    if spv_sections and spv_question:
        for quiz_name in selected_quizzes:
            sections.extend(build_spv_pdf_sections(
                response_df, quiz_name, spv_question, spv_part, spv_sections, colorblind_mode,
            ))

    if quiz_sections:
        sections.extend(build_quiz_pdf_sections(
            response_df, selected_quizzes, quiz_sections, colorblind_mode,
        ))

    return sections
