from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.latex_utils import clean_moodle_latex, extract_stack_answer_latex, maxima_expr_to_latex
from analytics.pdf_export import generate_pdf_report
from analytics.question_analytics import build_question_analytics
from analytics.question_details import build_error_drilldown, build_question_detail
from analytics.ui_theme import humanize_columns, inject_global_styles, pass_fail_scale, qualitative_colors
from analytics.upload_ui import inject_sidebar_css, load_shared_response_df, render_options_panel
from analytics.validation import audit_question_data


st.set_page_config(
    page_title="Question Analysis",
    page_icon=":bar_chart:",
    layout="wide",
)
inject_global_styles()

# Streamlit's own "⋮" menu (Deploy / theme / rerun, top-right) isn't extensible with
# custom entries, so the closest available "top-right corner" is a right-aligned
# toggle next to the page title.
title_col, colorblind_col = st.columns([5, 2])
with title_col:
    st.title("Question Analysis")
with colorblind_col:
    st.markdown("<div style='margin-top: 1.6rem;'></div>", unsafe_allow_html=True)
    colorblind_mode = st.toggle(
        "🎨 Colorblind Mode",
        key="colorblind_mode",
        help="Switches every chart (bars, box plots, scatter, line graphs, and the PRT pass-rate heatmap) to a red-green colorblind-safe palette.",
    )
st.warning("⏳ Depending on the size of your upload, it may take up to 30 seconds for all statistics to fully render, and up to 30 seconds for the downloadable PDF report to generate.")

inject_sidebar_css()

uploaded_files, anonymize_data = render_options_panel()


quiz_metadata, response_df, quiz_names = load_shared_response_df(uploaded_files, anonymize_data)
selected_quiz_name = None

# --- Sidebar: Question Analysis section (grouped separately so it can't get lost
# among the Quiz Analysis checkboxes below) ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Question Analysis")
st.sidebar.caption("Applies to the single quiz selected below")
if quiz_names:
    if len(quiz_names) > 1:
        selected_quiz_name = st.sidebar.selectbox("Select Quiz", quiz_names, index=0)
    else:
        selected_quiz_name = quiz_names[0]

_QUESTION_SECTION_KEYS = [
    "show_summary", "show_difficulty", "show_item_details",
    "show_response", "show_student", "show_metrics", "show_notes",
]
qa_select_col, qa_deselect_col = st.sidebar.columns(2)
# Setting session_state here, before the checkboxes below are instantiated in this
# same script run, is what makes the new value take effect immediately — a checkbox's
# `value=` argument is only its default the very first time a key is seen, so once a
# key exists in session_state (from either a prior user click or this button), the
# checkbox reads its state from there instead.
if qa_select_col.button("Select All", key="qa_select_all", use_container_width=True):
    for _key in _QUESTION_SECTION_KEYS:
        st.session_state[_key] = True
if qa_deselect_col.button("Deselect All", key="qa_deselect_all", use_container_width=True):
    for _key in _QUESTION_SECTION_KEYS:
        st.session_state[_key] = False

show_summary = st.sidebar.checkbox("1. Question Summary", value=True, key="show_summary")
show_difficulty = st.sidebar.checkbox("2. Question Difficulty Analysis", value=True, key="show_difficulty")
show_item_details = st.sidebar.checkbox("3. Question Item Details & Error Drill-Down", value=True, key="show_item_details")
show_response = st.sidebar.checkbox("4. Question Response Distribution", value=True, key="show_response")
show_student = st.sidebar.checkbox("5. Student Performance by Question", value=True, key="show_student")
show_metrics = st.sidebar.checkbox("6. Question Metrics", value=True, key="show_metrics")
show_notes = st.sidebar.checkbox("7. Interpretation Notes", value=True, key="show_notes")

if uploaded_files:
    if response_df.empty:
        st.info("No usable question rows were found in the uploaded files.")
    else:
        selected_df = response_df[response_df["quiz_name"] == selected_quiz_name].copy()
        analytics = build_question_analytics(selected_df, str(selected_quiz_name))

        st.success(f"Loaded {selected_quiz_name}" + (f" (1 of {len(quiz_names)} uploaded quizzes)" if len(quiz_names) > 1 else ""))

        question_summary = analytics["question_summary"]
        question_metrics = analytics["question_metrics"]
        response_outcomes = analytics["response_outcomes"]
        difficulty_metrics = analytics["difficulty_metrics"]
        syntax_analysis = analytics["syntax_analysis"]
        prt_pass_rates = analytics["prt_pass_rates"]
        repeated_wrong_answers = analytics["repeated_wrong_answers"]
        ranked_difficulty = analytics["ranked_difficulty"]
        export_summary = analytics["export_summary"]
        pool_a_df = analytics["pool_a_df"]
        pool_b_df = analytics["pool_b_df"]
        validation = audit_question_data(selected_df)

        if not isinstance(question_summary, dict):
            question_summary = dict(question_summary)
        if not isinstance(question_metrics, pd.DataFrame):
            question_metrics = pd.DataFrame(question_metrics)
        if not isinstance(response_outcomes, pd.DataFrame):
            response_outcomes = pd.DataFrame(response_outcomes)
        if not isinstance(difficulty_metrics, pd.DataFrame):
            difficulty_metrics = pd.DataFrame(difficulty_metrics)
        if not isinstance(syntax_analysis, pd.DataFrame):
            syntax_analysis = pd.DataFrame(syntax_analysis)
        if not isinstance(prt_pass_rates, pd.DataFrame):
            prt_pass_rates = pd.DataFrame(prt_pass_rates)
        if not isinstance(repeated_wrong_answers, pd.DataFrame):
            repeated_wrong_answers = pd.DataFrame(repeated_wrong_answers)
        if not isinstance(ranked_difficulty, pd.DataFrame):
            ranked_difficulty = pd.DataFrame(ranked_difficulty)

        # Student Performance Matrix - Pivoting Pool B over all questions (Q1..QN)
        pool_b_df["scaled_score"] = pool_b_df["grade"] * 10.0

        def _q_num(q_name: str) -> int:
            m = re.search(r"\d+", str(q_name))
            return int(m.group(0)) if m else 0

        question_order = sorted(pool_b_df["question"].unique(), key=_q_num)
        num_distinct_students = pool_b_df["student_id"].nunique()

        # pivot_table's default dropna=True silently drops any row/column that contains a
        # NaN anywhere. Use dropna=False + fill_value=0 so a single missing cell never
        # removes an entire student row or question column from the matrix.
        student_matrix = pool_b_df.pivot_table(
            index="student_id",
            columns="question",
            values="grade",
            aggfunc="first",
            fill_value=0.0,
            dropna=False,
        ).reindex(columns=question_order, fill_value=0.0)

        expected_shape = (num_distinct_students, len(question_order))
        if student_matrix.shape != expected_shape:
            st.warning(
                f"⚠️ Student performance matrix is {student_matrix.shape[0]} rows × {student_matrix.shape[1]} columns, "
                f"but {expected_shape[0]} students and {expected_shape[1]} questions were expected. "
                "Some student or question data may be missing upstream — check for NaN scores."
            )

        # Question Metrics Table
        metrics_flat = question_metrics.merge(
            difficulty_metrics[["question", "discrimination_index", "average_marks", "median_marks", "standard_deviation"]],
            on="question",
            how="left"
        )
        metrics_export = metrics_flat[[
            "question", "attempts", "students", "invalid_rate", "blank_rate",
            "reattempt_share", "facility", "partial_credit_mean",
            "discrimination_index", "average_marks", "median_marks", "standard_deviation",
            "catch_all_share"
        ]].rename(columns={"discrimination_index": "discrimination"})

        st.caption("The report below groups question-level analytics for the selected quiz into educational analysis areas. Cohort-level statistics combined across every uploaded file live on the Quiz Analysis page.")

        # 1. Question Summary Section
        if show_summary:
            with st.container(border=True):
                st.subheader("1. Question Summary")
                st.caption("This section summarises how each question was used, including participation, attempts, and whether responses could be interpreted successfully.")
                summary_metrics = [
                    ("Number of questions", question_summary["total_questions"]),
                    ("Number of students", question_summary["student_count"]),
                    ("Average score (out of 10)", f"{question_summary['average_score']:.2f}"),
                    ("Average valid submission rate", f"{question_summary['average_valid_submission_rate']:.2f}%"),
                    ("Average correct rate", f"{question_summary['average_correct_rate']:.2f}%"),
                    ("Syntax error count", question_summary["syntax_error_count"]),
                ]
                cols = st.columns(3)
                for index, (label, value) in enumerate(summary_metrics):
                    with cols[index % 3]:
                        st.metric(label, value)

                st.dataframe(
                    humanize_columns(
                        question_metrics[["question", "attempts", "students", "avg_score", "percent_valid", "percent_invalid", "syntax_error_count"]]
                        .rename(columns={"avg_score": "average_score"})
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        # 2. Question Difficulty Analysis Section
        if show_difficulty:
            with st.container(border=True):
                st.subheader("2. Question Difficulty Analysis")
                st.caption("This section evaluates how difficult each question was and how effectively it separates stronger from weaker students (sourced from Best Attempt per Student).")
                st.caption("⚠️ **Note on Discrimination (D)**: With small cohort sizes (around 30 students or fewer), the discrimination index is noisy and should be interpreted with caution.")

                st.dataframe(humanize_columns(ranked_difficulty.head(10)), use_container_width=True, hide_index=True)
                st.dataframe(humanize_columns(difficulty_metrics), use_container_width=True, hide_index=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        ranked_difficulty.head(10),
                        x="question",
                        y="avg_score",
                        color="question",
                        color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Set2),
                        labels={"avg_score": "Average score", "question": "Question"},
                    )
                    fig.update_layout(title="Top Difficult Questions by Average Score", showlegend=False, template="plotly")
                    st.plotly_chart(fig, use_container_width=True, key="difficulty_bar")
                with col2:
                    # Proper boxplot fed with Pool B per-student scores (same array as the
                    # Question Metrics table's average_marks/median_marks/standard_deviation).
                    fig2 = px.box(
                        pool_b_df,
                        x="question",
                        y="scaled_score",
                        color="question",
                        color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Set2),
                        labels={"scaled_score": "Score (0-10)", "question": "Question"},
                    )
                    fig2.update_layout(title="Score Distribution by Question (Best Attempt per Student)", showlegend=False, template="plotly")
                    st.plotly_chart(fig2, use_container_width=True, key="difficulty_box")

        # 3. Question Item Details & Error Drill-Down Section
        if show_item_details:
            with st.container(border=True):
                st.subheader("3. Question Item Details & Error Drill-Down")
                st.caption("Question text and the correct answer for each item, alongside where students went wrong (Best Attempt per Student). Populated only if the Moodle export included the Question text / Right answer Display options.")
                for q in question_order:
                    detail = build_question_detail(pool_b_df, q)
                    # Expander labels render as plain text (no Markdown/LaTeX/KaTeX at
                    # all) — showing cleaned-but-still-raw LaTeX there just displays the
                    # literal delimiters. Keep the collapsed title to the question number
                    # only; the full question text renders (as real math) once expanded.
                    question_text = clean_moodle_latex(detail["question_text"])
                    # Right Answer often carries the same "Seed: ...; ansN: <expr> [tag]"
                    # diagnostic dump as Submitted Response for STACK questions, so it
                    # gets the same ansN-extraction treatment (falls back to plain LaTeX
                    # cleanup when there's no ansN: pattern, e.g. a non-STACK quiz).
                    right_answer_text = extract_stack_answer_latex(detail["right_answer_text"])
                    drilldown = build_error_drilldown(pool_b_df, q)
                    with st.expander(f"Question {_q_num(q)}"):
                        st.markdown(f"**Question:** {question_text}")
                        st.markdown(f"**Right Answer:** {right_answer_text}")
                        if drilldown.empty:
                            st.success("No incorrect or partial-credit responses for this question among best attempts.")
                        else:
                            st.write(f"**Student Error Drill-Down** ({len(drilldown)} students didn't get full credit):")
                            st.dataframe(drilldown[["Student Name", "Email", "Score", "Status"]], use_container_width=True, hide_index=True)
                            st.caption("Submitted response vs. right answer (rendered as math where applicable):")
                            for _, row in drilldown.iterrows():
                                submitted = extract_stack_answer_latex(row["Submitted Response"])
                                right_answer = extract_stack_answer_latex(row["Right Answer"])
                                st.markdown(f"**{row['Student Name']}** — Submitted: {submitted}  \nRight Answer: {right_answer}")

        has_prt_data = bool(any(str(row.get("response_text", "")).strip() for _, row in selected_df.iterrows()))
        valid_invalid = pd.DataFrame({
            "question": question_metrics["question"],
            "Valid %": question_metrics["percent_valid"],
            "Invalid/Syntax Error %": question_metrics["percent_invalid"],
        })

        # 4. Question Response Distribution Section
        if show_response:
            with st.container(border=True):
                st.subheader("4. Question Response Distribution")
                st.caption("This section analyses how students answered each question, including common incorrect responses and potential misconceptions.")
                if not has_prt_data:
                    st.info("Upload a Responses file as well to see PRT/answer-note analysis for this quiz.")
                    st.dataframe(humanize_columns(response_outcomes), use_container_width=True, hide_index=True)
                    st.dataframe(humanize_columns(valid_invalid), use_container_width=True, hide_index=True)
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(
                            response_outcomes,
                            x="question",
                            y=["correct_percent", "incorrect_percent"],
                            barmode="group",
                            color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Vivid),
                            labels={"value": "Percent", "question": "Question"},
                        )
                        fig.update_layout(title="Response Outcome Percentages (Best Attempts)", template="plotly")
                        st.plotly_chart(fig, use_container_width=True, key="response_outcomes_bar")
                    with col2:
                        fig2 = px.bar(
                            valid_invalid,
                            x="question",
                            y=["Valid %", "Invalid/Syntax Error %"],
                            barmode="group",
                            color_discrete_sequence=qualitative_colors(colorblind_mode, px.colors.qualitative.Vivid),
                            labels={"value": "Percent", "question": "Question"},
                        )
                        fig2.update_layout(title="Valid vs Invalid Attempts (All Attempts)", template="plotly")
                        st.plotly_chart(fig2, use_container_width=True, key="response_validity_bar")

                    st.write("**Most Common Incorrect Answers** (rendered as math where applicable):")
                    with st.container(border=True):
                        header_cols = st.columns([1, 6])
                        header_cols[0].markdown("**Question**")
                        header_cols[1].markdown("**Most Common Incorrect Answers**")
                        st.divider()
                        for _, row in repeated_wrong_answers.iterrows():
                            top_wrong = row.get("top_wrong_expressions") or []
                            if top_wrong:
                                rendered = ", ".join(f"${maxima_expr_to_latex(expr)}$ ({cnt})" for expr, cnt in top_wrong)
                            else:
                                rendered = "None"
                            row_cols = st.columns([1, 6])
                            row_cols[0].markdown(f"**{row['question']}**")
                            row_cols[1].markdown(rendered)

                    if not prt_pass_rates.empty:
                        # dropna=False + fill_value=0 so a missing PRT pass-rate cell can't
                        # drop a whole question row or PRT column from the heatmap.
                        heatmap_df = prt_pass_rates.pivot_table(
                            index="question",
                            columns="prt_name",
                            values="pass_rate",
                            aggfunc="first",
                            fill_value=0,
                            dropna=False,
                        )
                        fig3 = px.imshow(
                            heatmap_df,
                            labels=dict(x="PRT", y="Question", color="Pass %"),
                            color_continuous_scale=pass_fail_scale(colorblind_mode),
                        )
                        # Explicit tick labels so Plotly can't thin out categorical ticks it deems crowded.
                        fig3.update_xaxes(tickmode="array", tickvals=list(range(len(heatmap_df.columns))), ticktext=[str(c) for c in heatmap_df.columns])
                        fig3.update_yaxes(tickmode="array", tickvals=list(range(len(heatmap_df.index))), ticktext=[str(r) for r in heatmap_df.index])
                        fig3.update_layout(title="PRT Pass Heatmap", template="plotly")
                        st.plotly_chart(fig3, use_container_width=True, key="prt_heatmap")
                    else:
                        st.info("No PRT pass data available for this quiz.")

        # 5. Student Performance by Question Section (renders all Q1..QN columns)
        if show_student:
            with st.container(border=True):
                st.subheader("5. Student Performance by Question")
                st.caption("This section compares student performance across questions to identify patterns of understanding (Best Attempt per Student).")
                st.dataframe(humanize_columns(student_matrix), use_container_width=True)
                fig = px.imshow(student_matrix, labels=dict(x="Question", y="Student", color="Score"), color_continuous_scale="Viridis")
                # Explicit tick labels on both axes so every question column and every
                # student row stays visible instead of Plotly thinning crowded ticks.
                fig.update_xaxes(tickmode="array", tickvals=list(range(len(student_matrix.columns))), ticktext=[str(c) for c in student_matrix.columns])
                fig.update_yaxes(tickmode="array", tickvals=list(range(len(student_matrix.index))), ticktext=[str(r) for r in student_matrix.index])
                # Scale the figure height to the student count so rows stay readable instead
                # of being squeezed into a fixed-height chart as the cohort grows.
                chart_height = max(400, 24 * len(student_matrix.index))
                fig.update_layout(title="Student-by-Question Performance Matrix (Best Attempts)", height=chart_height, template="plotly")
                st.plotly_chart(fig, use_container_width=True, key="student_matrix_heatmap")

        # 6. Question Metrics Section
        if show_metrics:
            with st.container(border=True):
                st.subheader("6. Question Metrics")
                st.caption("This section provides a consolidated numerical summary of every question-level metric and serves as the primary exportable dataset.")
                st.dataframe(humanize_columns(metrics_export), use_container_width=True, hide_index=True)
                st.caption("⚠️ **Note on Discrimination (D)**: With small cohort sizes (around 30 students or fewer), the discrimination index is noisy and should be interpreted with caution.")

        # 7. Interpretation Notes Section
        if show_notes:
            with st.container(border=True):
                st.subheader("7. Interpretation Notes & Export")
                st.caption("Use these notes to interpret the charts and export a styled PDF summary.")
                st.write("Validation summary:")
                st.write(validation["checks"])
                if validation["issues"]:
                    st.warning("\n".join(validation["issues"]))
                else:
                    st.success("The parsed Moodle response data passed the validation checks for the core analytics pipeline.")
                st.write("Interpretation guidance:")
                st.write("- Higher average scores indicate questions that were easier for the cohort.")
                st.write("- Lower score distributions and greater concentration of incorrect responses may indicate misconceptions or missing prerequisite knowledge.")
                st.write("- PRT pass rates help identify which branches of a Potential Response Tree are being routed correctly.")

        def _build_question_pdf_sections(quiz_name: str, selected_sections: list[str]) -> list[dict]:
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

                    if not q_prt_pass_rates.empty:
                        heatmap_df = q_prt_pass_rates.pivot_table(
                            index="question", columns="prt_name", values="pass_rate",
                            aggfunc="first", fill_value=0, dropna=False,
                        )
                        fig3 = px.imshow(
                            heatmap_df,
                            labels=dict(x="PRT", y="Question", color="Pass %"),
                            color_continuous_scale=pass_fail_scale(colorblind_mode),
                        )
                        fig3.update_xaxes(tickmode="array", tickvals=list(range(len(heatmap_df.columns))), ticktext=[str(c) for c in heatmap_df.columns])
                        fig3.update_yaxes(tickmode="array", tickvals=list(range(len(heatmap_df.index))), ticktext=[str(r) for r in heatmap_df.index])
                        fig3.update_layout(title="PRT Pass Heatmap", template="plotly")
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

        # PDF Report Options — scoped to PDF generation only; the on-screen sections
        # above are unaffected by these controls. Only sections currently enabled
        # on-screen (via the sidebar checkboxes) are offered here, since their
        # underlying tables/charts are only computed when that toggle is on.
        question_section_options = [
            (show_summary, "1. Question Summary"),
            (show_difficulty, "2. Question Difficulty Analysis"),
            (show_item_details, "3. Question Item Details & Error Drill-Down"),
            (show_response, "4. Question Response Distribution"),
            (show_student, "5. Student Performance Matrix"),
            (show_metrics, "6. Question Metrics"),
        ]
        available_question_sections = [label for enabled, label in question_section_options if enabled]

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### 📄 PDF Report Options")
            pdf_selected_question_sections = available_question_sections
            if available_question_sections:
                pdf_selected_question_sections = st.multiselect(
                    "Select which Question Analysis sections to include",
                    options=available_question_sections,
                    default=available_question_sections,
                )

            pdf_selected_quizzes = [selected_quiz_name]
            if len(quiz_names) > 1:
                pdf_selected_quizzes = st.multiselect(
                    "Select which quiz(zes) to include in the Question Analysis breakdown",
                    options=quiz_names,
                    default=[selected_quiz_name],
                )

        # Every selected quiz gets its own full 1-6 section run (via
        # _build_question_pdf_sections, in upload order) before the next quiz starts, so
        # e.g. 5 quizzes selected produces quiz1[1-6], quiz2[1-6], ..., quiz5[1-6] — not
        # all of section 1 across quizzes, then all of section 2, etc.
        pdf_sections = []
        quizzes_to_render = [q for q in quiz_names if q in pdf_selected_quizzes] or [selected_quiz_name]
        for quiz_name in quizzes_to_render:
            pdf_sections.extend(_build_question_pdf_sections(quiz_name, pdf_selected_question_sections))

        pdf_bytes = generate_pdf_report(
            title="Moodle STACK Question Analysis Report",
            subtitle=f"Quiz: {selected_quiz_name} • Generated Client-Side",
            sections=pdf_sections,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{selected_quiz_name}_question_analysis.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

else:
    # Pre-upload description & export guide
    with st.container(border=True):
        st.markdown("### 📊 Question Analysis")
        st.write("This section is for analyzing uploaded Moodle STACK quiz response files, one quiz at a time. Use the sidebar to upload one or more quiz responses files. After upload, you can:")
        st.markdown(
            """
            - review question summary metrics (attempts, students, invalid/blank rates, reattempts)
            - assess question difficulty and discrimination between stronger and weaker students
            - view each question's text and correct answer, with a drill-down of student errors
            - explore response distributions, PRT answer notes, and the most common wrong inputs
            - compare student performance across every question in the quiz
            - export a consolidated PDF report
            """
        )

        with st.container(border=True):
            st.markdown("<h5 style='margin-top:0;'>⚙️ Moodle Export Steps</h5>", unsafe_allow_html=True)
            st.markdown(
                """
                1️⃣ **Navigate to your target Quiz** in Moodle.<br>
                2️⃣ Open **Quiz results**.<br>
                3️⃣ Select **Responses report** from the Moodle report dropdown menu.<br>
                4️⃣ Under **Display options**, check the boxes for: **Question text**, **Response**, and **Right answer**.<br>
                5️⃣ Click **Display report**.<br>
                6️⃣ Download the generated report as a **CSV** or **XLSX** file.<br>
                7️⃣ Verify that your file contains the required structure below.
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("### 📦 Expected Data Format (Columns from Left to Right)")
            st.write("Your uploaded CSV or XLSX file must contain column headers ordered sequentially across the table:")
            st.markdown(
                """
                **1. Columns 1 to 8 (Student & Quiz Metadata):**
                `Last name` | `First name` | `Email address` | `State` | `Started on` | `Completed` | `Time taken` | `Grade/10.00`

                **2. Columns 9+ (Repeating Question Triplets):**
                - `Question 1` | `Response 1` | `Right answer 1`
                - `Question 2` | `Response 2` | `Right answer 2`
                - ...
                - `Question N` | `Response N` | `Right answer N`

                A few common alternate names are also recognized automatically, so exports using
                these instead will still work: `Username` (instead of `Email address`), `Status`
                (instead of `State`), `Started` (instead of `Started on`), and `Duration` (instead
                of `Time taken`).
                """
            )

        with st.container(border=True):
            c_text, c_btn = st.columns([3, 1])
            with c_text:
                st.markdown("**Want to try some sample data?**")
                st.write("Download pre-configured anonymized response reports to see the app in action or how your data should look.")
            with c_btn:
                st.link_button(
                    "📥 Sample Quiz Files",
                    url="https://drive.google.com/drive/folders/1r7c1asoMFwaLORaQVKisJk7xpWazzC5I?usp=sharing",
                    use_container_width=True,
                )

# Persistent Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: gray; margin-top: 1rem; margin-bottom: 1rem;">
        <div>Moodle/STACK Interactive Quiz Analytics is open-source and fully client-side.</div>
        <div>No quiz data is ever uploaded to external servers.</div>
    </div>
    """,
    unsafe_allow_html=True
)
