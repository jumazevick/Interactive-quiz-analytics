from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.difficulty import compute_difficulty_metrics
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
from analytics.upload_cache import CACHE_HASH_FUNCS, sync_uploaded_files
from analytics.question_metrics import compute_question_metrics, compute_question_summary, compute_ranked_difficulty
from analytics.response_analysis import compute_repeated_wrong_answers, compute_response_outcomes
from analytics.summary import build_export_summary
from analytics.syntax_analysis import compute_syntax_analysis
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
    page_title="Question Analysis Section",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Question Analysis Section")
st.header("Moodle/STACK Question Analytics")

# Sidebar - Options and Section Checkboxes (Part 5: Always visible before upload)
st.sidebar.title("Options")
uploaded_files = st.sidebar.file_uploader(
    "Upload responses file(s)",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True,
    help="Upload one or more Moodle responses exports in CSV, XLS, or XLSX format.",
)
uploaded_files, used_cached_upload = sync_uploaded_files(uploaded_files)
if used_cached_upload:
    st.sidebar.caption("📎 Using file(s) uploaded from the other Analysis section: " + ", ".join(f.name for f in uploaded_files))

st.sidebar.markdown("---")
st.sidebar.subheader("Visible Sections")
show_summary = st.sidebar.checkbox("Question Summary", value=True)
show_difficulty = st.sidebar.checkbox("Question Difficulty Analysis", value=True)
show_response = st.sidebar.checkbox("Question Response Distribution", value=True)
show_student = st.sidebar.checkbox("Student Performance by Question", value=True)
show_metrics = st.sidebar.checkbox("Question Metrics", value=True)
show_notes = st.sidebar.checkbox("Interpretation Notes", value=True)


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
                "completed_dt", "started_on", "attempt_idx", "source_type"
            ])

        if not parsed_df.empty:
            parsed_df["quiz_id"] = index
            parsed_df["quiz_name"] = quiz_name
            quiz_groups.setdefault(quiz_name, []).append(parsed_df)
        quiz_metadata.append({"quiz_id": index, "quiz_name": quiz_name})

    if not quiz_groups:
        return quiz_metadata, pd.DataFrame(columns=["student_id", "student_name", "question", "grade", "max_grade", "response_status", "response_text", "quiz_name", "quiz_id", "overall_grade", "completed_dt", "started_on", "attempt_idx", "source_type"])

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

        st.success(f"Loaded {selected_quiz_name}")

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

        # Part 3 Fix: Student Performance Matrix - Pivoting Pool B over all questions (Q1..QN)
        pool_b_df["scaled_score"] = pool_b_df["grade"] * 10.0

        def _q_num(q_name: str) -> int:
            m = re.search(r"\d+", str(q_name))
            return int(m.group(0)) if m else 0

        question_order = sorted(pool_b_df["question"].unique(), key=_q_num)
        num_distinct_students = pool_b_df["student_id"].nunique()

        # Round 4 fix: pivot_table's default dropna=True silently drops any row/column that
        # contains a NaN anywhere. Use dropna=False + fill_value=0 so a single missing cell
        # never removes an entire student row or question column from the matrix.
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

        # 4.5 Question Metrics Table
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

        st.caption("The report below groups question-level analytics into six educational analysis areas so the findings are easier to interpret.")

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
                        labels={"avg_score": "Average score", "question": "Question"},
                    )
                    fig.update_layout(title="Top Difficult Questions by Average Score")
                    st.plotly_chart(fig, use_container_width=True, key="difficulty_bar")
                    difficulty_section_charts.append({"title": "Top Difficult Questions by Average Score", "figure": fig})
                with col2:
                    # Part 2 Fix: Proper boxplot fed with Pool B per-student scores (same array as
                    # the Question Metrics table's average_marks/median_marks/standard_deviation).
                    fig2 = px.box(
                        pool_b_df,
                        x="question",
                        y="scaled_score",
                        labels={"scaled_score": "Score (0-10)", "question": "Question"},
                    )
                    fig2.update_layout(title="Score Distribution by Question (Best Attempt per Student)")
                    st.plotly_chart(fig2, use_container_width=True, key="difficulty_box")
                    difficulty_section_charts.append({"title": "Score Distribution by Question (Best Attempt per Student)", "figure": fig2})

        has_prt_data = bool(any(str(row.get("response_text", "")).strip() for _, row in selected_df.iterrows()))
        valid_invalid = pd.DataFrame({
            "question": question_metrics["question"],
            "Valid %": question_metrics["percent_valid"],
            "Invalid/Syntax Error %": question_metrics["percent_invalid"],
        })

        # 3. Question Response Distribution Section
        response_section_charts = []
        if show_response:
            with st.container(border=True):
                st.subheader("3. Question Response Distribution")
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
                            labels={"value": "Percent", "question": "Question"},
                        )
                        fig.update_layout(title="Response Outcome Percentages (Best Attempts)")
                        st.plotly_chart(fig, use_container_width=True, key="response_outcomes_bar")
                        response_section_charts.append({"title": "Response Outcome Percentages (Best Attempts)", "figure": fig})
                    with col2:
                        fig2 = px.bar(
                            valid_invalid,
                            x="question",
                            y=["Valid %", "Invalid/Syntax Error %"],
                            barmode="group",
                            labels={"value": "Percent", "question": "Question"},
                        )
                        fig2.update_layout(title="Valid vs Invalid Attempts (All Attempts)")
                        st.plotly_chart(fig2, use_container_width=True, key="response_validity_bar")
                        response_section_charts.append({"title": "Valid vs Invalid Attempts (All Attempts)", "figure": fig2})

                    st.dataframe(repeated_wrong_answers, use_container_width=True)

                    if not prt_pass_rates.empty:
                        # Round 4 fix: dropna=False + fill_value=0 so a missing PRT pass-rate
                        # cell can't drop a whole question row or PRT column from the heatmap.
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
                        fig3.update_layout(title="PRT Pass Heatmap")
                        st.plotly_chart(fig3, use_container_width=True, key="prt_heatmap")
                        response_section_charts.append({"title": "PRT Pass Heatmap", "figure": fig3})
                    else:
                        st.info("No PRT pass data available for this quiz.")


        # 4. Student Performance by Question Section (Part 3 Fix: Renders all Q1..QN columns)
        student_section_charts = []
        if show_student:
            with st.container(border=True):
                st.subheader("4. Student Performance by Question")
                st.caption("This section compares student performance across questions to identify patterns of understanding (Best Attempt per Student).")
                st.dataframe(student_matrix, use_container_width=True)
                fig = px.imshow(student_matrix, labels=dict(x="Question", y="Student", color="Score"), color_continuous_scale="Viridis")
                # Round 4 fix: explicit tick labels on both axes so every question column and
                # every student row stays visible instead of Plotly silently thinning out
                # categorical ticks it considers crowded.
                fig.update_xaxes(tickmode="array", tickvals=list(range(len(student_matrix.columns))), ticktext=[str(c) for c in student_matrix.columns])
                fig.update_yaxes(tickmode="array", tickvals=list(range(len(student_matrix.index))), ticktext=[str(r) for r in student_matrix.index])
                # Scale the figure height to the student count so rows stay readable instead of
                # being squeezed into a fixed-height chart as the cohort grows.
                chart_height = max(400, 24 * len(student_matrix.index))
                fig.update_layout(title="Student-by-Question Performance Matrix (Best Attempts)", height=chart_height)
                st.plotly_chart(fig, use_container_width=True, key="student_matrix_heatmap")
                student_section_charts.append({"title": "Student-by-Question Performance Matrix", "figure": fig})

        # 5. Question Metrics Section
        if show_metrics:
            with st.container(border=True):
                st.subheader("5. Question Metrics")
                st.caption("This section provides a consolidated numerical summary of every question-level metric and serves as the primary exportable dataset.")
                st.dataframe(metrics_export, use_container_width=True)
                st.caption("⚠️ **Note on Discrimination (D)**: With small cohort sizes (around 30 students or fewer), the discrimination index is noisy and should be interpreted with caution.")

        # 6. Interpretation Notes Section & PDF Export Button (Part 4)
        if show_notes:
            with st.container(border=True):
                st.subheader("6. Interpretation Notes & Export")
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

        # Single PDF Report Export Button (Part 4: tables + rendered chart images, same order as on-screen)
        pdf_sections = []
        if show_summary:
            pdf_sections.append({"title": "1. Question Summary", "caption": "Participation and summary statistics", "df": question_metrics[["question", "attempts", "students", "percent_valid", "percent_invalid", "syntax_error_count"]]})
        if show_difficulty:
            pdf_sections.append({"title": "2. Question Difficulty Analysis", "caption": "Facility and discrimination (Best Attempt)", "df": difficulty_metrics, "charts": difficulty_section_charts})
        if show_response:
            pdf_sections.append({"title": "3. Question Response Distribution", "caption": "Response outcomes and top wrong answers", "df": response_outcomes.merge(repeated_wrong_answers, on="question", how="left"), "charts": response_section_charts})
        if show_student:
            pdf_sections.append({"title": "4. Student Performance Matrix", "caption": "Per-student score per question (Best Attempt)", "df": student_matrix.reset_index(), "charts": student_section_charts})
        if show_metrics:
            pdf_sections.append({"title": "5. Question Metrics", "caption": "Consolidated question analytics table", "df": metrics_export})

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
    # Pre-upload description (Part 6.2)
    with st.container(border=True):
        st.markdown("### 📊 Question Analysis")
        st.write("This section is for analyzing uploaded Moodle STACK question-level response files. Use the sidebar to upload one or more quiz responses files. After upload, you can:")
        st.markdown(
            """
            - review question summary metrics (attempts, students, invalid/blank rates, reattempts)
            - assess question difficulty and discrimination between stronger and weaker students
            - explore response distributions, PRT answer notes, and the most common wrong inputs
            - compare student performance across every question in the quiz
            - export a consolidated, per-question metrics table
            """
        )
        st.write("If you need help downloading the files from Moodle, use the homepage guide.")

        # Shared Export Steps & Box B required columns
        with st.container(border=True):
            st.markdown("<h5 style='margin-top:0;'>⚙️ General Export Steps</h5>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("1️⃣ **Navigate to Your Quiz**\n\nClick into the specific STACK or standard Moodle quiz in your course workspace.")
                st.markdown("2️⃣ **Open Quiz Results**\n\nSelect 'Results' from the quiz secondary menu or settings menu.")
            with c2:
                st.markdown("3️⃣ **Choose Report Type (Responses)**\n\nSelect 'Responses' from the Moodle report dropdown.")
                st.markdown("4️⃣ **Download Table Data**\n\nScroll to the bottom of the page, select Comma Separated Values (.csv) or Microsoft Excel (.xlsx), and click 'Download'.")

        with st.container(border=True):
            st.markdown("📦 **B. Question & PRT Analysis Required Columns**")
            st.markdown(
                """
                ```python
                // Identical metadata columns:
                - Surname, First name, Email address, State, Started on, Completed, Time taken, Grade/10.00
                // Plus response columns:
                - Response 1
                - Response 2
                - Response 3
                - ...
                - Response N (matches your quiz count)
                ```
                """
            )

# Persistent Footer (Part 6.2)
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


