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
        analytics = build_question_analytics(selected_df, selected_quiz_name)

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

        st.metric("Total Questions", question_summary["total_questions"])
        st.metric("Overall PRT Elements", question_summary["overall_prt_elements"])
        st.metric("Most Difficult Question", question_summary["most_difficult_question"])
        st.metric("Syntax Error Count", question_summary["syntax_error_count"])
        st.metric("Average Score", f"{question_summary['average_score']:.2f}")
        st.metric("Average Valid Submission Rate", f"{question_summary['average_valid_submission_rate']:.2f}%")
        st.metric("Average Correct Rate", f"{question_summary['average_correct_rate']:.2f}%")
        st.metric("Number of Students", question_summary["student_count"])

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

        st.subheader("Top Most Difficult Questions")
        st.dataframe(ranked_difficulty.head(10))

        st.subheader("Question Metrics Summary")
        st.dataframe(question_metrics[["question", "attempts", "avg_score", "percent_correct", "percent_incorrect", "percent_valid", "percent_invalid", "syntax_error_count", "syntax_error_percent"]].rename(columns={"avg_score": "average_score"}))

        st.subheader("PRT Pass Heatmap")
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

        st.subheader("Repeated Wrong Answer Trends")
        st.dataframe(repeated_wrong_answers)

        st.subheader("Difficulty Metrics")
        st.dataframe(difficulty_metrics)

        st.subheader("Syntax Error Analysis")
        st.dataframe(syntax_analysis)

        st.subheader("Overall Question Analytics Summary")
        st.download_button(
            label="Download summary CSV",
            data=export_summary.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_quiz_name}_question_analysis.csv",
            mime="text/csv",
        )
        st.dataframe(export_summary)
else:
    st.info("Upload one or more question results files to begin.")
