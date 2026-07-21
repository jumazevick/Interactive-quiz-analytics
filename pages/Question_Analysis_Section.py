from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.difficulty import compute_difficulty_metrics
from analytics.parser import build_response_rows, parse_uploaded_file
from analytics.prt_analysis import build_prt_frame, compute_prt_pass_rates
from analytics.question_metrics import compute_question_metrics, compute_question_summary, compute_ranked_difficulty
from analytics.response_analysis import compute_response_outcomes, compute_repeated_wrong_answers
from analytics.summary import build_export_summary
from analytics.syntax_analysis import compute_syntax_analysis
from analytics.validation import audit_question_data


def build_question_analytics(response_df: pd.DataFrame, quiz_name: str) -> dict[str, object]:
    if "question" not in response_df.columns:
        response_df = build_response_rows(response_df, quiz_name=quiz_name)

    question_metrics = compute_question_metrics(response_df)
    prt_frame = build_prt_frame(response_df)
    question_summary = compute_question_summary(response_df, prt_frame)
    response_outcomes = compute_response_outcomes(response_df)
    difficulty_metrics = compute_difficulty_metrics(response_df)
    syntax_analysis = compute_syntax_analysis(response_df)
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
    }


st.set_page_config(
    page_title="Question Analysis Section",
    page_icon=":bar_chart:",
)

st.title("Question Analysis Section")
st.header("Question-level Analysis")

st.sidebar.title("Options")
uploaded_files = st.sidebar.file_uploader(
    "Upload question results file(s)",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True,
    help="Upload one or more Moodle question results exports in CSV, XLS, or XLSX format.",
)


@st.cache_data(show_spinner=False)
def load_quiz_data(files) -> tuple[list[dict[str, object]], pd.DataFrame]:
    parsed_frames: list[pd.DataFrame] = []
    quiz_metadata: list[dict[str, object]] = []
    for index, uploaded_file in enumerate(files, start=1):
        df = parse_uploaded_file(uploaded_file)
        response_df = build_response_rows(df, quiz_name=uploaded_file.name)
        if not response_df.empty:
            response_df["quiz_id"] = index
            response_df["quiz_name"] = uploaded_file.name
            parsed_frames.append(response_df)
        quiz_metadata.append({"quiz_id": index, "quiz_name": uploaded_file.name})

    if not parsed_frames:
        return quiz_metadata, pd.DataFrame(columns=["student_id", "student_name", "question", "grade", "max_grade", "response_status", "response_text", "quiz_name", "quiz_id"])

    combined = pd.concat(parsed_frames, ignore_index=True)
    return quiz_metadata, combined

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
        if not isinstance(export_summary, pd.DataFrame):
            export_summary = pd.DataFrame(export_summary)

        st.caption("The report below groups question-level analytics into six educational analysis areas so the findings are easier to interpret.")

        with st.container():
            st.subheader("1. Question Summary")
            st.caption("This section summarises how each question was used, including participation, attempts, and whether responses could be interpreted successfully.")
            summary_metrics = [
                ("Number of questions", question_summary["total_questions"]),
                ("Number of students", question_summary["student_count"]),
                ("Average score", f"{question_summary['average_score']:.2f}"),
                ("Average valid submission rate", f"{question_summary['average_valid_submission_rate']:.2f}%"),
                ("Average correct rate", f"{question_summary['average_correct_rate']:.2f}%"),
                ("Syntax error count", question_summary["syntax_error_count"]),
            ]
            cols = st.columns(3)
            for index, (label, value) in enumerate(summary_metrics):
                with cols[index % 3]:
                    st.metric(label, value)

            st.dataframe(question_metrics[["question", "attempts", "avg_score", "percent_valid", "percent_invalid", "syntax_error_count"]].rename(columns={"avg_score": "average_score"}))

        with st.container():
            st.subheader("2. Question Difficulty Analysis")
            st.caption("This section evaluates how difficult each question was and how effectively it separates stronger from weaker students.")
            st.dataframe(ranked_difficulty.head(10))
            st.dataframe(difficulty_metrics)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    ranked_difficulty.head(10),
                    x="question",
                    y="average_score",
                    labels={"average_score": "Average score", "question": "Question"},
                )
                fig.update_layout(title="Top Difficult Questions by Average Score")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.box(question_metrics, x="question", y="scaled_score", labels={"scaled_score": "Scaled score", "question": "Question"})
                fig2.update_layout(title="Score Distribution by Question")
                st.plotly_chart(fig2, use_container_width=True)

        with st.container():
            st.subheader("3. Question Response Distribution")
            st.caption("This section analyses how students answered each question, including common incorrect responses and potential misconceptions.")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    response_outcomes,
                    x="question",
                    y=["correct_percent", "incorrect_percent"],
                    barmode="group",
                    labels={"value": "Percent", "question": "Question"},
                )
                fig.update_layout(title="Response Outcome Percentages")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                valid_invalid = pd.DataFrame({
                    "question": question_metrics["question"],
                    "Valid %": question_metrics["percent_valid"],
                    "Invalid/Syntax Error %": question_metrics["percent_invalid"],
                })
                fig2 = px.bar(
                    valid_invalid,
                    x="question",
                    y=["Valid %", "Invalid/Syntax Error %"],
                    barmode="group",
                    labels={"value": "Percent", "question": "Question"},
                )
                fig2.update_layout(title="Valid vs Invalid Attempts")
                st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(repeated_wrong_answers)

            if not prt_pass_rates.empty:
                heatmap_df = prt_pass_rates.pivot(index="question", columns="prt_name", values="pass_rate").fillna(0)
                fig3 = px.imshow(
                    heatmap_df,
                    labels=dict(x="PRT", y="Question", color="Pass %"),
                    color_continuous_scale=["#ef4444", "#fde68a", "#22c55e"],
                )
                fig3.update_layout(title="PRT Pass Heatmap")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No PRT pass data available for this quiz.")

        with st.container():
            st.subheader("4. Student Performance by Question")
            st.caption("This section compares student performance across questions to identify patterns of understanding and areas requiring further support.")
            student_matrix = selected_df.pivot_table(index="student_id", columns="question", values="grade", aggfunc="mean").fillna(0)
            st.dataframe(student_matrix)
            fig = px.imshow(student_matrix, labels=dict(x="Question", y="Student", color="Score"), color_continuous_scale="Viridis")
            fig.update_layout(title="Student-by-Question Performance Matrix")
            st.plotly_chart(fig, use_container_width=True)

        with st.container():
            st.subheader("5. Question Metrics")
            st.caption("This section provides a consolidated numerical summary of every question-level metric and serves as the primary exportable dataset.")
            st.dataframe(question_metrics[["question", "attempts", "avg_score", "percent_correct", "percent_incorrect", "percent_valid", "percent_invalid", "syntax_error_count", "syntax_error_percent", "scaled_score"]].rename(columns={"avg_score": "average_score"}))

        with st.container():
            st.subheader("6. Interpretation Notes")
            st.caption("Use these notes to interpret the charts, understand the metric definitions, and review caveats before making instructional decisions.")
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
            st.download_button(
                label="Download full question analytics summary",
                data=export_summary.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_quiz_name}_question_analysis.csv",
                mime="text/csv",
            )
            st.markdown("For the full methodology and definitions, refer to the provided Question Analytics document in your project materials.")
else:
    st.info("Upload one or more question results files to begin.")
