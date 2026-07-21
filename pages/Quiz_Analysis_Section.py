import re
from datetime import datetime, timedelta

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from analytics.summary import create_excel_report


st.set_page_config(page_title="Quiz Analysis Section", page_icon=":chart_with_upwards_trend:", layout="wide")

df_filtered = pd.DataFrame()


def parse_time(time_str):
    regex = re.compile(
        r"((?P<weeks>\d+?) ?weeks?)? ?"
        r"((?P<days>\d+?) ?days?)? ?"
        r"((?P<hours>\d+?) ?hours?)? ?"
        r"((?P<minutes>\d+?) ?mins?)? ?"
        r"((?P<seconds>\d+?) ?secs?)?"
    )
    parts = regex.match(time_str)
    if parts is None:
        return None
    parts = parts.groupdict()
    time_params = {name: int(param) for name, param in parts.items() if param}
    return timedelta(**time_params).total_seconds()


def parse_datetime(value):
    try:
        return datetime.strptime(value, "%d %B %Y %I:%M %p")
    except ValueError:
        return None


def normalize_grades(df):
    grade_column_pattern = re.compile(r"Grade/\d+(?:\.\d+)?")
    grade_column = None
    for column in df.columns:
        if grade_column_pattern.match(column):
            grade_column = column
            break

    if grade_column is None:
        raise KeyError("Grade column not found in the dataset")

    max_grade_value = float(re.search(r"\d+(?:\.\d+)?", grade_column).group())
    df[grade_column] = pd.to_numeric(df[grade_column], errors="coerce")

    if max_grade_value != 10:
        df["Normalized_Grade"] = (df[grade_column] / max_grade_value) * 10
    else:
        df["Normalized_Grade"] = df[grade_column]

    return df


def load_data(uploaded_files):
    dfs = []
    for quiz_id, file in enumerate(uploaded_files):
        if file.name.endswith(".xls"):
            df = pd.read_excel(file, engine="xlrd")
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file, engine="openpyxl")
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            raise ValueError("Unsupported file format: " + file.name)

        df.insert(3, "quizID", quiz_id + 1, True)
        if "Last name" in df.columns:
            df = df.rename(columns={"Last name": "Surname"})

        df = normalize_grades(df)

        if "Email address" in df.columns and df["Email address"].notna().any():
            student_id = df["Email address"]
        elif "anonymized_full_name" in df.columns:
            student_id = df["anonymized_full_name"]
        else:
            student_id = pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)

        if "anonymized_full_name" in df.columns:
            student_id = student_id.fillna(df["anonymized_full_name"])

        student_id = student_id.fillna(pd.Series([f"row_{i}" for i in range(len(df))], index=df.index))
        df["student_id"] = student_id.astype(str)

        columns = [
            "Surname",
            "First name",
            "Email address",
            "student_id",
            "quizID",
            "State",
            "Started on",
            "Completed",
            "Time taken",
            "Normalized_Grade",
        ]
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns: {', '.join(missing_columns)}")

        dfs.append(df[columns])

    df = pd.concat(dfs)
    df = df[df.State == "Finished"]
    df["Started on"] = df["Started on"].apply(parse_datetime)
    df["Completed"] = df["Completed"].apply(parse_datetime)
    df["Time taken"] = df["Time taken"].apply(parse_time)
    df = df.rename(
        columns={
            "Surname": "surname",
            "First name": "firstname",
            "Email address": "email",
            "student_id": "student_id",
            "State": "state",
            "Started on": "start_date",
            "Completed": "end_date",
            "Time taken": "time_taken",
            "Normalized_Grade": "grade",
        }
    )
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    return df


st.title("Quiz Analysis Section")
st.header("Moodle STACK Analytics")

st.markdown(
    """
This section is for analyzing uploaded Moodle STACK quiz files.

Use the sidebar to upload one or more quiz result files. After upload, you can:

- review merged attempts from multiple quizzes
- compare student count, attempt rate, mean grade, and grade spread
- view grade distributions
- check engagement over time
- compare attempts against grades

If you need help downloading the files from Moodle, use the homepage guide.
"""
)

st.sidebar.title("Options")

uploaded_files = st.sidebar.file_uploader(
    "Upload grades file(s)",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True,
    help="Upload one or more Moodle grades exports in CSV, XLS, or XLSX format.",
)

if uploaded_files:
    df = load_data(uploaded_files)
    if not df.empty:
        selected_quizzes = st.sidebar.multiselect(
            "Select Quiz IDs",
            options=df["quizID"].unique(),
            default=df["quizID"].unique(),
        )
        df_filtered = df[df["quizID"].isin(selected_quizzes)]


def generate_quiz_stats(selected_stats):
    if df_filtered.empty:
        return pd.DataFrame()

    dfs = []

    if "student_count" in selected_stats:
        students_per_quiz = df_filtered.groupby("quizID")["student_id"].nunique().reset_index()
        students_per_quiz.columns = ["quizID", "student_count"]
        dfs.append(students_per_quiz)

    if "mean_grade" in selected_stats or "grade_variance" in selected_stats:
        grade_statistics = df_filtered.groupby("quizID")["grade"].agg(["mean", "var"]).reset_index()
        grade_statistics.columns = ["quizID", "mean_grade", "grade_variance"]
        if "mean_grade" not in selected_stats:
            grade_statistics = grade_statistics.drop(columns=["mean_grade"])
        if "grade_variance" not in selected_stats:
            grade_statistics = grade_statistics.drop(columns=["grade_variance"])
        dfs.append(grade_statistics)

    if "mean_highest_grade" in selected_stats:
        highest_grades = df_filtered.groupby(["quizID", "student_id"])["grade"].max().reset_index()
        average_highest_grades = highest_grades.groupby("quizID")["grade"].mean().reset_index()
        average_highest_grades.columns = ["quizID", "mean_highest_grade"]
        dfs.append(average_highest_grades)

    if "attempt_count" in selected_stats:
        total_attempts_per_quiz = df_filtered.groupby("quizID").size().reset_index(name="attempt_count")
        dfs.append(total_attempts_per_quiz)

    if "attempt_rate" in selected_stats:
        attempts_per_student = df_filtered.groupby(["quizID", "student_id"]).size().reset_index(name="attempt_count")
        average_attempts_per_student = attempts_per_student.groupby("quizID")["attempt_count"].mean().reset_index()
        average_attempts_per_student.columns = ["quizID", "attempt_rate"]
        dfs.append(average_attempts_per_student)

    if dfs:
        quiz_stats = dfs[0]
        for frame in dfs[1:]:
            quiz_stats = pd.merge(quiz_stats, frame, on="quizID")
        return quiz_stats.round(2)
    return pd.DataFrame()


# Export Excel Report Setup
if not df_filtered.empty:
    all_stats_cols = ["student_count", "attempt_rate", "mean_grade", "grade_variance", "mean_highest_grade", "attempt_count"]
    quiz_stats_excel = generate_quiz_stats(all_stats_cols)
    excel_data = {
        "Merged Attempts": df_filtered,
        "Quiz Statistics": quiz_stats_excel,
    }
    excel_bytes = create_excel_report(excel_data)
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Download Excel Report",
        data=excel_bytes,
        file_name="quiz_analysis_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


if st.sidebar.checkbox("Merged List of Users and Files"):
    with st.container(border=True):
        st.write("### Merged List of Users and Files")
        st.write(
            "This table combines all uploaded quiz files into one view. Each row is one attempt, with the student, quiz, date, time taken, and normalized grade."
        )
        st.dataframe(df_filtered, use_container_width=True)

show_summary = st.sidebar.checkbox("Summary of Quiz Stats")

if show_summary:
    with st.container(border=True):
        st.write("### Summary of Quiz Statistics")
        if not df_filtered.empty:
            selected_stats = st.sidebar.multiselect(
                "Select Statistics to Display",
                [
                    "student_count",
                    "attempt_rate",
                    "mean_grade",
                    "grade_variance",
                    "mean_highest_grade",
                    "attempt_count",
                ],
                default=[
                    "student_count",
                    "attempt_rate",
                    "mean_grade",
                    "grade_variance",
                    "mean_highest_grade",
                    "attempt_count",
                ],
            )

            summary_notes = {
                "student_count": "- student_count: how many different students attempted each quiz.\n",
                "attempt_rate": "- attempt_rate: the average number of attempts per student for each quiz.\n",
                "mean_grade": "- mean_grade: the average score for each quiz.\n",
                "grade_variance": "- grade_variance: how spread out the grades are for each quiz.\n",
                "mean_highest_grade": "- mean_highest_grade: the average best score each student reached in a quiz.\n",
                "attempt_count": "- attempt_count: the total number of attempts made on each quiz.\n",
            }

            summary_text = " ".join([summary_notes[stat] for stat in selected_stats])
            st.write(summary_text)
            quiz_stats = generate_quiz_stats(selected_stats)
            st.dataframe(quiz_stats, use_container_width=True)
        else:
            st.write("You need to upload a file(s) to initiate the analysis.")

if st.sidebar.checkbox("Quiz Grade Distribution (Box plot)", False):
    if not df_filtered.empty:
        with st.container(border=True):
            st.write("### Quiz Grade Distribution (Box plot)")
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 6))

            st.write(
                """
- Box Plot: shows how grades are spread out for each quiz.
- Median: the green line marks the middle value of grades for each quiz.
- mean_grade line: the red line shows the average grade for each quiz.
"""
            )

            sns.boxplot(x="quizID", y="grade", data=df_filtered, palette="Set3", medianprops=dict(color="#00FF00"))
            sns.stripplot(x="quizID", y="grade", data=df_filtered, color="black", jitter=0.1, size=1.8)

            means = df_filtered.groupby("quizID")["grade"].mean().reset_index()
            plt.plot(
                means["quizID"].astype(int) - 1,
                means["grade"],
                marker="o",
                color="#FF474C",
                linestyle="-",
                linewidth=2,
                label="mean_grade",
            )

            plt.title("Grade Distribution")
            plt.xlabel("Quiz ID")
            plt.ylabel("Grade")
            plt.legend()
            st.pyplot(plt)
    else:
        st.info("You need to upload a file(s) to initiate the analysis.")

if st.sidebar.checkbox("Frquency Density (Engagement) "):
    with st.container(border=True):
        st.write("### Engagement Over Time")
        if "start_date" in list(df_filtered.columns):
            df_filtered["start_date"] = pd.to_datetime(df_filtered["start_date"])
            plt.figure(figsize=(10, 8))

            if not df_filtered["start_date"].isnull().any():
                for quiz_id in df_filtered["quizID"].unique():
                    quiz_data = df_filtered[df_filtered["quizID"] == quiz_id]
                    if not quiz_data.empty:
                        sns.kdeplot(quiz_data["start_date"], label=f"Quiz {quiz_id}")
                    else:
                        st.write(f"No data for Quiz ID: {quiz_id}")

                plt.title("Engagement Over Time")
                plt.xlabel("Date")
                plt.ylabel("Frequency Density")
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)

                st.write(
                    """
This graph shows how students participation in quizzes changes over time.
The line graph displays the frequency of quiz attempts on different dates.
Each line represents a different quiz, showing when students started taking it.

What You Can Learn:
- Trends: see if there are certain times when more students are attempting quizzes.
- Peak Times: identify dates when engagement is high or low for each quiz.
- Comparisons: compare the participation trends across different quizzes.
"""
                )
            else:
                st.write("No valid dates available for plotting.")

if st.sidebar.checkbox("Scatter plot: Attempts vs Grades"):
    if "quizID" in list(df_filtered.columns):
        with st.container(border=True):
            st.write("### Scatter plot: Attempts vs Grades")
            grade_type = st.sidebar.radio("Select Grade Type", ("Highest Grade", "Average Grade", "Minimum Grade"))
            plt.figure(figsize=(25, 15))

            quiz_attempt_count = df_filtered.groupby(["quizID", "student_id"]).size().reset_index(name="attempt_count")

            if grade_type == "Highest Grade":
                grade_data = df_filtered.groupby(["quizID", "student_id"])["grade"].max().reset_index()
                y_label = "Highest Grade"
                title = "Attempts vs Highest Grade"
            elif grade_type == "Minimum Grade":
                grade_data = df_filtered.groupby(["quizID", "student_id"])["grade"].min().reset_index()
                y_label = "Minimum Grade"
                title = "Attempts vs Minimum Grade"
            else:
                grade_data = df_filtered.groupby(["quizID", "student_id"])["grade"].mean().reset_index()
                y_label = "Average Grade"
                title = "Attempts vs Average Grade"

            merged_data = pd.merge(quiz_attempt_count, grade_data, on=["quizID", "student_id"])
            correlation = merged_data["attempt_count"].corr(merged_data["grade"])
            st.write(f"Correlation between Attempts and Quiz {y_label}: r = {correlation:.2f}")

            sns.scatterplot(data=merged_data, x="attempt_count", y="grade", hue="quizID", palette="husl", marker="o", s=200)
            plt.title(title, fontsize=20)
            plt.xlabel("No. of Attempts", fontsize=20)
            plt.ylabel(y_label, fontsize=20)
            plt.legend(title="Quiz ID", fontsize=20)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            st.pyplot(plt)

            st.write(
                f"""
Attempts vs {y_label}:

This scatter plot shows how the number of attempts relates to the {y_label} for each quiz.

- Plot: each point represents a student's attempts and their {y_label}.
- Highest Grade: the best score a student achieved.
- Minimum Grade: the lowest score a student achieved.
- Average Grade: the overall performance across attempts.
"""
            )
    else:
        st.info("You need to upload a file(s) to initiate the analysis.")

if st.sidebar.checkbox("Line Graph of Various Metrics"):
    with st.container(border=True):
        st.write("### Line Graph of Various Metrics")
        selected_metrics = st.sidebar.multiselect(
            "Select Metrics to Display",
            ["student_count", "attempt_rate", "mean_grade", "grade_variance"],
            default=["student_count", "attempt_rate", "mean_grade", "grade_variance"],
        )

        if selected_metrics:
            if "student_count" in selected_metrics:
                st.write("student_count: number of unique students who attempted each quiz.")
            if "attempt_rate" in selected_metrics:
                st.write("attempt_rate: average number of attempts per student for each quiz.")
            if "mean_grade" in selected_metrics:
                st.write("mean_grade: average score for each quiz.")
            if "grade_variance" in selected_metrics:
                st.write("grade_variance: how spread out the grades are for each quiz.")

            def generate_line_graph_data(frame, metrics):
                data = {}
                if "student_count" in metrics:
                    data["student_count"] = frame.groupby("quizID")["student_id"].nunique()
                if "attempt_rate" in metrics:
                    attempts_per_student = frame.groupby(["quizID", "student_id"]).size().reset_index(name="attempt_count")
                    data["attempt_rate"] = attempts_per_student.groupby("quizID")["attempt_count"].mean()
                if "mean_grade" in metrics:
                    data["mean_grade"] = frame.groupby("quizID")["grade"].mean()
                if "grade_variance" in metrics:
                    data["grade_variance"] = frame.groupby("quizID")["grade"].var()
                return pd.DataFrame(data).reset_index()

            if not df_filtered.empty:
                line_graph_data = generate_line_graph_data(df_filtered, selected_metrics)
                line_graph_data = line_graph_data.melt("quizID", var_name="Metric", value_name="Value")

                line_chart = alt.Chart(line_graph_data).mark_line().encode(
                    x=alt.X("quizID:O", title="Quiz ID"),
                    y=alt.Y("Value:Q"),
                    color="Metric:N",
                    tooltip=["quizID", "Metric", "Value"],
                ).properties(width=600, height=400)

                points = line_chart.mark_point().encode(
                    x=alt.X("quizID:O", title="Quiz ID"),
                    y=alt.Y("Value:Q"),
                    color="Metric:N",
                    tooltip=["quizID", "Metric", "Value"],
                )

                st.altair_chart(line_chart + points, use_container_width=True)
        else:
            st.info("You need to upload a file(s) to initiate the analysis.")

# Persistent Footer (Part 6.2)
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: gray; margin-top: 1rem; margin-bottom: 1rem;">
        <div>Moodle STACK Analytics Hub is open-source and fully client-side.</div>
        <div>No quiz data is ever uploaded to external servers.</div>
    </div>
    """,
    unsafe_allow_html=True
)

