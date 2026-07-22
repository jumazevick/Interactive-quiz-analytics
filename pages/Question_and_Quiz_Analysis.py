from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.difficulty import compute_difficulty_metrics
from analytics.latex_utils import format_moodle_latex
from analytics.parser import (
    build_grade_breakdown_rows,
    build_response_rows,
    detect_export_type,
    get_attempt_pools,
    merge_grade_breakdown_rows,
    parse_uploaded_file,
)
from analytics.pdf_export import generate_pdf_report
from analytics.prt_analysis import build_prt_frame, compute_prt_pass_rates
from analytics.question_details import build_error_drilldown, build_question_detail
from analytics.question_metrics import compute_question_metrics, compute_question_summary, compute_ranked_difficulty
from analytics.quiz_metrics import (
    build_boxplot_figure,
    build_engagement_figure,
    build_line_graph_figure,
    build_metric_trend_data,
    build_quiz_attempt_frame,
    build_scatter_figure,
    compute_quiz_stats,
)
from analytics.response_analysis import compute_repeated_wrong_answers, compute_response_outcomes
from analytics.summary import build_export_summary
from analytics.syntax_analysis import compute_syntax_analysis
from analytics.upload_cache import CACHE_HASH_FUNCS, clear_uploaded_files, get_uploader_key, sync_uploaded_files
from analytics.validation import audit_question_data


def build_question_analytics(response_df: pd.DataFrame, quiz_name: str) -> dict[str, object]:
    if "question" not in response_df.columns:
        export_type = detect_export_type(response_df)
        if export_type == "grades_breakdown":
            response_df = build_grade_breakdown_rows(response_df, quiz_name=quiz_name)
        else:
            response_df = build_response_rows(response_df, quiz_name=quiz_name)

    pool_a_df, pool_b_df = get_attempt_pools(response_df)

    question_metrics = compute_question_metrics(response_df)
    prt_frame = build_prt_frame(pool_a_df)
    question_summary = compute_question_summary(response_df, prt_frame)
    response_outcomes = compute_response_outcomes(response_df)
    difficulty_metrics = compute_difficulty_metrics(response_df)
    syntax_analysis = compute_syntax_analysis(pool_a_df)
    prt_pass_rates = compute_prt_pass_rates(prt_frame)
    repeated_wrong_answers = compute_repeated_wrong_answers(response_df)
    ranked_difficulty = compute_ranked_difficulty(question_metrics)
    export_summary = build_export_summary(question_metrics, response_outcomes, difficulty_metrics, syntax_analysis, prt_pass_rates, repeated_wrong_answers)

    return {
        "question_metrics": question_metrics,
        "question_summary": question_summary,
        "response_outcomes": response_outcomes,
        "difficulty_metrics": difficulty_metrics,
        "syntax_analysis": syntax_analysis,
        "prt_frame": prt_frame,
        "prt_pass_rates": prt_pass_rates,
        "repeated_wrong_answers": repeated_wrong_answers,
        "ranked_difficulty": ranked_difficulty,
        "export_summary": export_summary,
        "quiz_name": quiz_name,
        "pool_a_df": pool_a_df,
        "pool_b_df": pool_b_df,
    }


st.set_page_config(
    page_title="Question & Quiz Analysis",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Question & Quiz Analysis")
st.header("Moodle/STACK Question & Quiz Analytics")

# Sidebar - Options and Section Checkboxes (always visible before upload)
st.sidebar.title("Options")
uploaded_files = st.sidebar.file_uploader(
    "Upload responses file(s)",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True,
    help="Upload one or more Moodle responses exports in CSV, XLS, or XLSX format.",
    key=get_uploader_key(),
)
uploaded_files, used_cached_upload = sync_uploaded_files(uploaded_files)
if used_cached_upload:
    st.sidebar.caption("📎 Using previously uploaded file(s): " + ", ".join(f.name for f in uploaded_files))

if st.sidebar.button("🗑️ Clear / Reset All Uploaded Files", use_container_width=True):
    clear_uploaded_files()
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Visible Sections")
st.sidebar.caption("Question Analysis (selected quiz)")
show_summary = st.sidebar.checkbox("1. Question Summary", value=True)
show_difficulty = st.sidebar.checkbox("2. Question Difficulty Analysis", value=True)
show_item_details = st.sidebar.checkbox("3. Question Item Details & Error Drill-Down", value=True)
show_response = st.sidebar.checkbox("4. Question Response Distribution", value=True)
show_student = st.sidebar.checkbox("5. Student Performance by Question", value=True)
show_metrics = st.sidebar.checkbox("6. Question Metrics", value=True)
show_notes = st.sidebar.checkbox("7. Interpretation Notes", value=True)

st.sidebar.caption("Quiz Analysis (combined across all uploaded files)")
show_quiz_merged = st.sidebar.checkbox("8. Merged List of Users and Files")
show_quiz_summary = st.sidebar.checkbox("9. Summary of Quiz Stats")
show_quiz_boxplot = st.sidebar.checkbox("10. Quiz Grade Distribution (Box Plot)")
show_quiz_engagement = st.sidebar.checkbox("11. Engagement Over Time")
show_quiz_scatter = st.sidebar.checkbox("12. Scatter Plot: Attempts vs Grades")
show_quiz_linegraph = st.sidebar.checkbox("13. Line Graph of Various Metrics")


@st.cache_data(show_spinner=False, hash_funcs=CACHE_HASH_FUNCS)
def load_quiz_data(files) -> tuple[list[dict[str, object]], pd.DataFrame]:
    quiz_groups: dict[str, list[pd.DataFrame]] = {}
    quiz_metadata: list[dict[str, object]] = []

    def normalize_quiz_name(file_name: str) -> str:
        name = file_name.rsplit(".", 1)[0]
        return re.sub(r"[-_](responses|grades|grade)$", "", name, flags=re.IGNORECASE)

    for index, uploaded_file in enumerate(files, start=1):
        df = parse_uploaded_file(uploaded_file)
        export_type = detect_export_type(df)
        quiz_name = normalize_quiz_name(uploaded_file.name)
        if export_type == "grades_breakdown":
            parsed_df = build_grade_breakdown_rows(df, quiz_name=quiz_name)
        elif export_type == "responses":
            parsed_df = build_response_rows(df, quiz_name=quiz_name)
        else:
            parsed_df = pd.DataFrame(columns=[
                "student_id", "student_name", "question", "grade", "max_grade",
                "response_status", "response_text", "quiz_name", "overall_grade",
                "completed_dt", "started_on", "attempt_idx", "source_type",
                "question_text", "right_answer_text"
            ])

        if not parsed_df.empty:
            parsed_df["quiz_id"] = index
            parsed_df["quiz_name"] = quiz_name
            quiz_groups.setdefault(quiz_name, []).append(parsed_df)
        quiz_metadata.append({"quiz_id": index, "quiz_name": quiz_name})

    if not quiz_groups:
        return quiz_metadata, pd.DataFrame(columns=["student_id", "student_name", "question", "grade", "max_grade", "response_status", "response_text", "quiz_name", "quiz_id", "overall_grade", "completed_dt", "started_on", "attempt_idx", "source_type", "question_text", "right_answer_text"])

    combined_frames = []
    for quiz_name, frames in quiz_groups.items():
        combined = pd.concat(frames, ignore_index=True)
        if len(frames) > 1:
            response_frames = [frame for frame in frames if not frame.empty and frame.get("source_type", "responses").eq("responses").any()]
            grade_frames = [frame for frame in frames if not frame.empty and frame.get("source_type", "responses").eq("grades_breakdown").any()]
            if response_frames and grade_frames:
                response_rows = pd.concat(response_frames, ignore_index=True)
                grade_rows = pd.concat(grade_frames, ignore_index=True)
                combined = merge_grade_breakdown_rows(response_rows, grade_rows)
                combined["quiz_name"] = quiz_name
                combined["quiz_id"] = 0
        combined_frames.append(combined)

    return quiz_metadata, pd.concat(combined_frames, ignore_index=True)


if uploaded_files:
    quiz_metadata, response_df = load_quiz_data(uploaded_files)
    if response_df.empty:
        st.info("No usable question rows were found in the uploaded files.")
    else:
        quiz_names = [item["quiz_name"] for item in quiz_metadata]
        if len(quiz_names) > 1:
            selected_quiz_name = st.sidebar.selectbox("Select Quiz", quiz_names, index=0)
        else:
            selected_quiz_name = quiz_names[0]

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

        st.caption("The report below groups question-level analytics into educational analysis areas, followed by a combined Quiz Analysis section across every uploaded file.")

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
                    question_metrics[["question", "attempts", "students", "avg_score", "percent_valid", "percent_invalid", "syntax_error_count"]]
                    .rename(columns={"avg_score": "average_score"}),
                    use_container_width=True,
                )

        # 2. Question Difficulty Analysis Section
        difficulty_section_charts = []
        if show_difficulty:
            with st.container(border=True):
                st.subheader("2. Question Difficulty Analysis")
                st.caption("This section evaluates how difficult each question was and how effectively it separates stronger from weaker students (sourced from Best Attempt per Student).")
                st.caption("⚠️ **Note on Discrimination (D)**: With small cohort sizes (around 30 students or fewer), the discrimination index is noisy and should be interpreted with caution.")

                st.dataframe(ranked_difficulty.head(10), use_container_width=True)
                st.dataframe(difficulty_metrics, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        ranked_difficulty.head(10),
                        x="question",
                        y="avg_score",
                        color="question",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        labels={"avg_score": "Average score", "question": "Question"},
                    )
                    fig.update_layout(title="Top Difficult Questions by Average Score", showlegend=False, template="plotly")
                    st.plotly_chart(fig, use_container_width=True, key="difficulty_bar")
                    difficulty_section_charts.append({"title": "Top Difficult Questions by Average Score", "figure": fig})
                with col2:
                    # Proper boxplot fed with Pool B per-student scores (same array as the
                    # Question Metrics table's average_marks/median_marks/standard_deviation).
                    fig2 = px.box(
                        pool_b_df,
                        x="question",
                        y="scaled_score",
                        color="question",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        labels={"scaled_score": "Score (0-10)", "question": "Question"},
                    )
                    fig2.update_layout(title="Score Distribution by Question (Best Attempt per Student)", showlegend=False, template="plotly")
                    st.plotly_chart(fig2, use_container_width=True, key="difficulty_box")
                    difficulty_section_charts.append({"title": "Score Distribution by Question (Best Attempt per Student)", "figure": fig2})

        # 3. Question Item Details & Error Drill-Down Section
        item_details_rows = []
        if show_item_details:
            with st.container(border=True):
                st.subheader("3. Question Item Details & Error Drill-Down")
                st.caption("Question text and the correct answer for each item, alongside where students went wrong (Best Attempt per Student). Populated only if the Moodle export included the Question text / Right answer Display options.")
                for q in question_order:
                    detail = build_question_detail(pool_b_df, q)
                    question_text = format_moodle_latex(detail["question_text"])
                    right_answer_text = format_moodle_latex(detail["right_answer_text"])
                    drilldown = build_error_drilldown(pool_b_df, q)
                    with st.expander(f"{q}: {question_text[:80]}"):
                        st.markdown(f"**Question:** {question_text}")
                        st.markdown(f"**Right Answer:** {right_answer_text}")
                        if drilldown.empty:
                            st.success("No incorrect or partial-credit responses for this question among best attempts.")
                        else:
                            st.write(f"**Student Error Drill-Down** ({len(drilldown)} students didn't get full credit):")
                            st.dataframe(drilldown[["Student Name", "Email", "Score", "Status"]], use_container_width=True)
                            st.caption("Submitted response vs. right answer (rendered as math where applicable):")
                            for _, row in drilldown.iterrows():
                                submitted = format_moodle_latex(row["Submitted Response"])
                                right_answer = format_moodle_latex(row["Right Answer"])
                                st.markdown(f"**{row['Student Name']}** — Submitted: {submitted}  \nRight Answer: {right_answer}")
                            flat = drilldown.copy()
                            flat.insert(0, "Question", q)
                            flat["Submitted Response"] = flat["Submitted Response"].apply(format_moodle_latex)
                            flat["Right Answer"] = flat["Right Answer"].apply(format_moodle_latex)
                            item_details_rows.append(flat)

        item_details_pdf_table = pd.concat(item_details_rows, ignore_index=True) if item_details_rows else pd.DataFrame()

        has_prt_data = bool(any(str(row.get("response_text", "")).strip() for _, row in selected_df.iterrows()))
        valid_invalid = pd.DataFrame({
            "question": question_metrics["question"],
            "Valid %": question_metrics["percent_valid"],
            "Invalid/Syntax Error %": question_metrics["percent_invalid"],
        })

        # 4. Question Response Distribution Section
        response_section_charts = []
        if show_response:
            with st.container(border=True):
                st.subheader("4. Question Response Distribution")
                st.caption("This section analyses how students answered each question, including common incorrect responses and potential misconceptions.")
                if not has_prt_data:
                    st.info("Upload a Responses file as well to see PRT/answer-note analysis for this quiz.")
                    st.dataframe(response_outcomes, use_container_width=True)
                    st.dataframe(valid_invalid, use_container_width=True)
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(
                            response_outcomes,
                            x="question",
                            y=["correct_percent", "incorrect_percent"],
                            barmode="group",
                            color_discrete_sequence=px.colors.qualitative.Vivid,
                            labels={"value": "Percent", "question": "Question"},
                        )
                        fig.update_layout(title="Response Outcome Percentages (Best Attempts)", template="plotly")
                        st.plotly_chart(fig, use_container_width=True, key="response_outcomes_bar")
                        response_section_charts.append({"title": "Response Outcome Percentages (Best Attempts)", "figure": fig})
                    with col2:
                        fig2 = px.bar(
                            valid_invalid,
                            x="question",
                            y=["Valid %", "Invalid/Syntax Error %"],
                            barmode="group",
                            color_discrete_sequence=px.colors.qualitative.Vivid,
                            labels={"value": "Percent", "question": "Question"},
                        )
                        fig2.update_layout(title="Valid vs Invalid Attempts (All Attempts)", template="plotly")
                        st.plotly_chart(fig2, use_container_width=True, key="response_validity_bar")
                        response_section_charts.append({"title": "Valid vs Invalid Attempts (All Attempts)", "figure": fig2})

                    st.dataframe(repeated_wrong_answers, use_container_width=True)

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
                            color_continuous_scale=["#ef4444", "#fde68a", "#22c55e"],
                        )
                        # Explicit tick labels so Plotly can't thin out categorical ticks it deems crowded.
                        fig3.update_xaxes(tickmode="array", tickvals=list(range(len(heatmap_df.columns))), ticktext=[str(c) for c in heatmap_df.columns])
                        fig3.update_yaxes(tickmode="array", tickvals=list(range(len(heatmap_df.index))), ticktext=[str(r) for r in heatmap_df.index])
                        fig3.update_layout(title="PRT Pass Heatmap", template="plotly")
                        st.plotly_chart(fig3, use_container_width=True, key="prt_heatmap")
                        response_section_charts.append({"title": "PRT Pass Heatmap", "figure": fig3})
                    else:
                        st.info("No PRT pass data available for this quiz.")

        # 5. Student Performance by Question Section (renders all Q1..QN columns)
        student_section_charts = []
        if show_student:
            with st.container(border=True):
                st.subheader("5. Student Performance by Question")
                st.caption("This section compares student performance across questions to identify patterns of understanding (Best Attempt per Student).")
                st.dataframe(student_matrix, use_container_width=True)
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
                student_section_charts.append({"title": "Student-by-Question Performance Matrix", "figure": fig})

        # 6. Question Metrics Section
        if show_metrics:
            with st.container(border=True):
                st.subheader("6. Question Metrics")
                st.caption("This section provides a consolidated numerical summary of every question-level metric and serves as the primary exportable dataset.")
                st.dataframe(metrics_export, use_container_width=True)
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

        st.markdown("<br>", unsafe_allow_html=True)
        st.header("Quiz Analysis")
        st.caption(f"Combined across all {len(quiz_names)} uploaded quiz file(s), independent of the quiz selected above.")

        # 8-13. Quiz Analysis (combined across every uploaded quiz file)
        attempt_frame = build_quiz_attempt_frame(response_df)

        quiz_merged_table = None
        quiz_summary_table = None
        quiz_boxplot_fig = None
        quiz_engagement_fig = None
        quiz_scatter_fig = None
        quiz_linegraph_fig = None

        if show_quiz_merged:
            with st.container(border=True):
                st.subheader("8. Merged List of Users and Files")
                st.caption("Combines every uploaded quiz file into one view. Each row is one attempt, with the student, quiz, and date.")
                st.dataframe(attempt_frame, use_container_width=True)
                quiz_merged_table = attempt_frame

        if show_quiz_summary:
            with st.container(border=True):
                st.subheader("9. Summary of Quiz Stats")
                st.caption("Aggregated statistics per quiz, combined across all uploaded files.")
                if not attempt_frame.empty:
                    selected_quiz_stats = st.sidebar.multiselect(
                        "Select Statistics to Display",
                        ["student_count", "attempt_rate", "mean_grade", "grade_variance", "mean_highest_grade", "attempt_count"],
                        default=["student_count", "attempt_rate", "mean_grade", "grade_variance", "mean_highest_grade", "attempt_count"],
                    )
                    quiz_stats_df = compute_quiz_stats(attempt_frame, selected_quiz_stats)
                    st.dataframe(quiz_stats_df, use_container_width=True)
                    quiz_summary_table = quiz_stats_df
                else:
                    st.info("No quiz attempt data available yet.")

        if show_quiz_boxplot:
            with st.container(border=True):
                st.subheader("10. Quiz Grade Distribution (Box Plot)")
                st.caption("Spread of grades per quiz, with mean grade overlay, combined across all uploaded files.")
                if not attempt_frame.empty:
                    fig = build_boxplot_figure(attempt_frame)
                    fig.update_layout(template="plotly")
                    st.plotly_chart(fig, use_container_width=True, key="quiz_boxplot")
                    quiz_boxplot_fig = fig
                else:
                    st.info("No quiz attempt data available yet.")

        if show_quiz_engagement:
            with st.container(border=True):
                st.subheader("11. Engagement Over Time")
                st.caption("Density of quiz attempt start times per quiz, combined across all uploaded files.")
                if not attempt_frame.empty:
                    fig = build_engagement_figure(attempt_frame)
                    if fig is not None:
                        fig.update_layout(template="plotly")
                        st.plotly_chart(fig, use_container_width=True, key="quiz_engagement")
                        quiz_engagement_fig = fig
                    else:
                        st.info("Not enough date variation across attempts to estimate an engagement density.")
                else:
                    st.info("No quiz attempt data available yet.")

        if show_quiz_scatter:
            with st.container(border=True):
                st.subheader("12. Scatter Plot: Attempts vs Grades")
                st.caption("Correlation between number of attempts and grade outcome, combined across all uploaded files.")
                if not attempt_frame.empty:
                    grade_type = st.sidebar.radio("Select Grade Type", ("Highest Grade", "Average Grade", "Minimum Grade"))
                    result = build_scatter_figure(attempt_frame, grade_type)
                    if result is not None:
                        fig, correlation, y_label, _ = result
                        fig.update_layout(template="plotly")
                        st.write(f"Correlation between Attempts and Quiz {y_label}: r = {correlation:.2f}")
                        st.plotly_chart(fig, use_container_width=True, key="quiz_scatter")
                        quiz_scatter_fig = fig
                else:
                    st.info("No quiz attempt data available yet.")

        if show_quiz_linegraph:
            with st.container(border=True):
                st.subheader("13. Line Graph of Various Metrics")
                st.caption("Trend of selected metrics across quizzes, combined across all uploaded files.")
                if not attempt_frame.empty:
                    selected_quiz_metrics = st.sidebar.multiselect(
                        "Select Metrics to Display",
                        ["student_count", "attempt_rate", "mean_grade", "grade_variance"],
                        default=["student_count", "attempt_rate", "mean_grade", "grade_variance"],
                    )
                    if selected_quiz_metrics:
                        trend_data = build_metric_trend_data(attempt_frame, selected_quiz_metrics)
                        fig = build_line_graph_figure(trend_data)
                        fig.update_layout(template="plotly")
                        st.plotly_chart(fig, use_container_width=True, key="quiz_linegraph")
                        quiz_linegraph_fig = fig
                else:
                    st.info("No quiz attempt data available yet.")

        def _build_question_pdf_sections(quiz_name: str) -> list[dict]:
            """Core Question Analysis PDF sections (summary, difficulty w/ charts, response
            distribution, student matrix) for any uploaded quiz — lets the PDF include a
            breakdown for quizzes other than the one currently shown on-screen."""
            quiz_df = response_df[response_df["quiz_name"] == quiz_name].copy()
            if quiz_df.empty:
                return []

            quiz_analytics = build_question_analytics(quiz_df, str(quiz_name))
            q_metrics = quiz_analytics["question_metrics"]
            q_difficulty = quiz_analytics["difficulty_metrics"]
            q_response_outcomes = quiz_analytics["response_outcomes"]
            q_repeated_wrong = quiz_analytics["repeated_wrong_answers"]
            q_pool_b = quiz_analytics["pool_b_df"].copy()
            q_ranked_difficulty = quiz_analytics["ranked_difficulty"]

            if q_pool_b.empty:
                return []

            q_pool_b["scaled_score"] = q_pool_b["grade"] * 10.0
            q_order = sorted(q_pool_b["question"].unique(), key=_q_num)
            prefix = f"[{quiz_name}] "

            sections = [{
                "title": f"{prefix}Question Summary",
                "caption": "Participation and summary statistics",
                "df": q_metrics[["question", "attempts", "students", "percent_valid", "percent_invalid", "syntax_error_count"]],
            }]

            difficulty_charts = []
            fig = px.bar(
                q_ranked_difficulty.head(10), x="question", y="avg_score", color="question",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"avg_score": "Average score", "question": "Question"},
            )
            fig.update_layout(title="Top Difficult Questions by Average Score", showlegend=False, template="plotly")
            difficulty_charts.append({"title": "Top Difficult Questions by Average Score", "figure": fig})

            fig2 = px.box(
                q_pool_b, x="question", y="scaled_score", color="question",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"scaled_score": "Score (0-10)", "question": "Question"},
            )
            fig2.update_layout(title="Score Distribution by Question (Best Attempt per Student)", showlegend=False, template="plotly")
            difficulty_charts.append({"title": "Score Distribution by Question (Best Attempt per Student)", "figure": fig2})

            sections.append({
                "title": f"{prefix}Question Difficulty Analysis",
                "caption": "Facility and discrimination (Best Attempt)",
                "df": q_difficulty,
                "charts": difficulty_charts,
            })

            sections.append({
                "title": f"{prefix}Question Response Distribution",
                "caption": "Response outcomes and top wrong answers",
                "df": q_response_outcomes.merge(q_repeated_wrong, on="question", how="left"),
            })

            student_matrix_q = q_pool_b.pivot_table(
                index="student_id", columns="question", values="grade",
                aggfunc="first", fill_value=0.0, dropna=False,
            ).reindex(columns=q_order, fill_value=0.0)
            sections.append({
                "title": f"{prefix}Student Performance Matrix",
                "caption": "Per-student score per question (Best Attempt)",
                "df": student_matrix_q.reset_index(),
            })

            return sections

        # PDF Report Options — scoped to PDF generation only; the on-screen sections
        # above are unaffected by these controls.
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### 📄 PDF Report Options")
            pdf_include_quiz_summary = st.checkbox("Include Quiz Analysis Summary", value=True)
            pdf_include_question_breakdown = st.checkbox("Include Question Analysis Breakdown", value=True)

            pdf_selected_quizzes = [selected_quiz_name]
            if pdf_include_question_breakdown and len(quiz_names) > 1:
                pdf_selected_quizzes = st.multiselect(
                    "Select which quiz(zes) to include in the Question Analysis breakdown",
                    options=quiz_names,
                    default=[selected_quiz_name],
                )

        # Single combined PDF Report Export Button (tables + rendered chart images, same
        # order as on-screen), gated by the same sidebar checkboxes as what's visible,
        # plus the PDF-only scope controls above.
        pdf_sections = []
        if pdf_include_question_breakdown:
            if show_summary:
                pdf_sections.append({"title": "1. Question Summary", "caption": f"Participation and summary statistics ({selected_quiz_name})", "df": question_metrics[["question", "attempts", "students", "percent_valid", "percent_invalid", "syntax_error_count"]]})
            if show_difficulty:
                pdf_sections.append({"title": "2. Question Difficulty Analysis", "caption": "Facility and discrimination (Best Attempt)", "df": difficulty_metrics, "charts": difficulty_section_charts})
            if show_item_details:
                pdf_sections.append({"title": "3. Question Item Details & Error Drill-Down", "caption": "Question text, right answer, and wrong-response drill-down (Best Attempt)", "df": item_details_pdf_table})
            if show_response:
                pdf_sections.append({"title": "4. Question Response Distribution", "caption": "Response outcomes and top wrong answers", "df": response_outcomes.merge(repeated_wrong_answers, on="question", how="left"), "charts": response_section_charts})
            if show_student:
                pdf_sections.append({"title": "5. Student Performance Matrix", "caption": "Per-student score per question (Best Attempt)", "df": student_matrix.reset_index(), "charts": student_section_charts})
            if show_metrics:
                pdf_sections.append({"title": "6. Question Metrics", "caption": "Consolidated question analytics table", "df": metrics_export})

            for extra_quiz in pdf_selected_quizzes:
                if extra_quiz == selected_quiz_name:
                    continue
                pdf_sections.extend(_build_question_pdf_sections(extra_quiz))

        if pdf_include_quiz_summary:
            if show_quiz_merged and quiz_merged_table is not None:
                pdf_sections.append({"title": "8. Merged List of Users and Files", "caption": "All parsed quiz attempt rows (combined across uploaded files)", "df": quiz_merged_table})
            if show_quiz_summary and quiz_summary_table is not None:
                pdf_sections.append({"title": "9. Summary of Quiz Stats", "caption": "Aggregated stats per quiz", "df": quiz_summary_table})
            if show_quiz_boxplot and quiz_boxplot_fig is not None:
                pdf_sections.append({"title": "10. Quiz Grade Distribution (Box Plot)", "caption": "Spread of grades per quiz, with mean grade overlay", "charts": [{"title": "Grade Distribution", "figure": quiz_boxplot_fig}]})
            if show_quiz_engagement and quiz_engagement_fig is not None:
                pdf_sections.append({"title": "11. Engagement Over Time", "caption": "Density of quiz attempt start times per quiz", "charts": [{"title": "Engagement Over Time", "figure": quiz_engagement_fig}]})
            if show_quiz_scatter and quiz_scatter_fig is not None:
                pdf_sections.append({"title": "12. Scatter Plot: Attempts vs Grades", "caption": "Correlation between number of attempts and grade outcome", "charts": [{"title": "Attempts vs Grades", "figure": quiz_scatter_fig}]})
            if show_quiz_linegraph and quiz_linegraph_fig is not None:
                pdf_sections.append({"title": "13. Line Graph of Various Metrics", "caption": "Trend of selected metrics across quizzes", "charts": [{"title": "Metrics by Quiz", "figure": quiz_linegraph_fig}]})

        pdf_bytes = generate_pdf_report(
            title="Moodle STACK Question & Quiz Analysis Report",
            subtitle=f"Quiz: {selected_quiz_name} • {len(quiz_names)} quiz file(s) combined • Generated Client-Side",
            sections=pdf_sections,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{selected_quiz_name}_question_and_quiz_analysis.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

else:
    # Pre-upload description & export guide
    with st.container(border=True):
        st.markdown("### 📊 Question & Quiz Analysis")
        st.write("This section is for analyzing uploaded Moodle STACK quiz response files. Use the sidebar to upload one or more quiz responses files. After upload, you can:")
        st.markdown(
            """
            - review question summary metrics (attempts, students, invalid/blank rates, reattempts)
            - assess question difficulty and discrimination between stronger and weaker students
            - view each question's text and correct answer, with a drill-down of student errors
            - explore response distributions, PRT answer notes, and the most common wrong inputs
            - compare student performance across every question in the quiz
            - review cohort-level quiz stats, grade distributions, and engagement trends combined across every uploaded file
            - export a single, consolidated PDF report
            """
        )
        st.write("If you need help downloading the files from Moodle, use the homepage guide.")

        with st.container(border=True):
            st.markdown("<h5 style='margin-top:0;'>⚙️ Moodle Export Steps</h5>", unsafe_allow_html=True)
            st.markdown(
                """
                1️⃣ **Navigate to your target Quiz** in Moodle.
                2️⃣ Open **Quiz results**.
                3️⃣ Select **Responses report** from the Moodle report dropdown menu.
                4️⃣ Under **Display options**, check the boxes for: **Question text**, **Response**, and **Right answer**.
                5️⃣ Click **Display report**.
                6️⃣ Download the generated report as a **CSV** or **XLSX** file.
                7️⃣ Verify that your file contains the required structure below.
                """
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
                """
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
