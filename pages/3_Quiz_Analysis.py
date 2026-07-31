from __future__ import annotations

import streamlit as st

from analytics.pdf_ui import render_pdf_report_panel
from analytics.quiz_metrics import (
    build_boxplot_figure,
    build_engagement_figure,
    build_line_graph_figure,
    build_metric_trend_data,
    build_quiz_attempt_frame,
    build_scatter_figure,
    compute_quiz_stats,
)
from analytics.ui_theme import humanize_column_name, humanize_columns, inject_global_styles
from analytics.upload_ui import inject_sidebar_css, load_shared_response_df, render_options_panel, render_sidebar_bottom_spacer


st.set_page_config(
    page_title="Quiz Analysis",
    page_icon=":bar_chart:",
    layout="wide",
)
inject_global_styles()

# Streamlit's own "⋮" menu (Deploy / theme / rerun, top-right) isn't extensible with
# custom entries, so the closest available "top-right corner" is a right-aligned
# toggle next to the page title.
title_col, colorblind_col = st.columns([5, 2])
with title_col:
    st.title("Quiz Analysis")
with colorblind_col:
    st.markdown("<div style='margin-top: 1.6rem;'></div>", unsafe_allow_html=True)
    colorblind_mode = st.toggle(
        "🎨 Colorblind Mode",
        key="quiz_colorblind_mode",
        help="Switches every chart (bars, box plots, scatter, line graphs, and the PRT pass-rate heatmap) to a red-green colorblind-safe palette.",
    )
st.warning("⏳ Depending on the size of your upload, it may take up to 30 seconds for all statistics to fully render, and up to 30 seconds for the downloadable PDF report to generate.")

inject_sidebar_css()

uploaded_files, anonymize_data = render_options_panel()


quiz_metadata, response_df, quiz_names = load_shared_response_df(uploaded_files, anonymize_data)

# --- Sidebar: Quiz Analysis section — each sub-widget lives right under the
# checkbox/section it configures, instead of being scattered wherever its section
# happens to render in the main body. ---
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Quiz Analysis")
st.sidebar.caption("Combined across the quizzes selected below")
quizzes_for_analysis = quiz_names
if quiz_names:
    quizzes_for_analysis = st.sidebar.multiselect(
        "Select quizzes to include",
        options=quiz_names,
        default=quiz_names,
    )
_QUIZ_SECTION_KEYS = [
    "show_quiz_merged", "show_quiz_summary", "show_quiz_boxplot",
    "show_quiz_engagement", "show_quiz_scatter", "show_quiz_linegraph",
]
quiz_select_col, quiz_deselect_col = st.sidebar.columns(2)
if quiz_select_col.button("Select All", key="quiz_select_all", use_container_width=True):
    for _key in _QUIZ_SECTION_KEYS:
        st.session_state[_key] = True
if quiz_deselect_col.button("Deselect All", key="quiz_deselect_all", use_container_width=True):
    for _key in _QUIZ_SECTION_KEYS:
        st.session_state[_key] = False

show_quiz_merged = st.sidebar.checkbox("1. Merged List of Users and Files", value=True, key="show_quiz_merged")
show_quiz_summary = st.sidebar.checkbox("2. Summary of Quiz Stats", value=True, key="show_quiz_summary")
selected_quiz_stats: list[str] = []
if show_quiz_summary:
    selected_quiz_stats = st.sidebar.multiselect(
        "Select Statistics to Display",
        ["student_count", "attempt_rate", "mean_grade", "grade_variance", "mean_highest_grade", "attempt_count"],
        default=["student_count", "attempt_rate", "mean_grade", "grade_variance", "mean_highest_grade", "attempt_count"],
        format_func=humanize_column_name,
    )
show_quiz_boxplot = st.sidebar.checkbox("3. Quiz Grade Distribution (Box Plot)", value=True, key="show_quiz_boxplot")
show_quiz_engagement = st.sidebar.checkbox("4. Engagement Over Time", value=True, key="show_quiz_engagement")
show_quiz_scatter = st.sidebar.checkbox("5. Scatter Plot: Attempts vs Grades", value=True, key="show_quiz_scatter")
quiz_grade_type = "Average Grade"
if show_quiz_scatter:
    quiz_grade_type = st.sidebar.radio("Select Grade Type", ("Highest Grade", "Average Grade", "Minimum Grade"))
show_quiz_linegraph = st.sidebar.checkbox("6. Line Graph of Various Metrics", value=True, key="show_quiz_linegraph")
selected_quiz_metrics: list[str] = []
if show_quiz_linegraph:
    selected_quiz_metrics = st.sidebar.multiselect(
        "Select Metrics to Display",
        ["student_count", "attempt_rate", "mean_grade", "grade_variance"],
        default=["student_count", "attempt_rate", "mean_grade", "grade_variance"],
        format_func=humanize_column_name,
    )

render_sidebar_bottom_spacer()

if uploaded_files:
    if response_df.empty:
        st.info("No usable question rows were found in the uploaded files.")
    else:
        st.caption(f"Combined across {len(quizzes_for_analysis)} of {len(quiz_names)} uploaded quiz file(s) — see the quiz selector in the sidebar.")

        # 8-13. Quiz Analysis (combined across the quizzes selected in the sidebar)
        attempt_frame = build_quiz_attempt_frame(response_df[response_df["quiz_name"].isin(quizzes_for_analysis)])

        if show_quiz_merged:
            with st.container(border=True):
                st.subheader("1. Merged List of Users and Files")
                st.caption("Combines every uploaded quiz file into one view. Each row is one attempt, with the student, quiz, and date.")
                st.dataframe(humanize_columns(attempt_frame), use_container_width=True, hide_index=True)

        if show_quiz_summary:
            with st.container(border=True):
                st.subheader("2. Summary of Quiz Stats")
                st.caption("Aggregated statistics per quiz, combined across all uploaded files.")
                if not attempt_frame.empty:
                    quiz_stats_df = compute_quiz_stats(attempt_frame, selected_quiz_stats)
                    st.dataframe(humanize_columns(quiz_stats_df), use_container_width=True, hide_index=True)
                else:
                    st.info("No quiz attempt data available yet.")

        if show_quiz_boxplot:
            with st.container(border=True):
                st.subheader("3. Quiz Grade Distribution (Box Plot)")
                st.caption("Spread of grades per quiz, with mean grade overlay, combined across all uploaded files.")
                if not attempt_frame.empty:
                    fig = build_boxplot_figure(attempt_frame, colorblind_mode=colorblind_mode)
                    fig.update_layout(template="plotly")
                    st.plotly_chart(fig, use_container_width=True, key="quiz_boxplot")
                else:
                    st.info("No quiz attempt data available yet.")

        if show_quiz_engagement:
            with st.container(border=True):
                st.subheader("4. Engagement Over Time")
                st.caption("Density of quiz attempt start times per quiz, combined across all uploaded files.")
                if not attempt_frame.empty:
                    fig = build_engagement_figure(attempt_frame, colorblind_mode=colorblind_mode)
                    if fig is not None:
                        fig.update_layout(template="plotly")
                        st.plotly_chart(fig, use_container_width=True, key="quiz_engagement")
                    else:
                        st.info("Not enough date variation across attempts to estimate an engagement density.")
                else:
                    st.info("No quiz attempt data available yet.")

        if show_quiz_scatter:
            with st.container(border=True):
                st.subheader("5. Scatter Plot: Attempts vs Grades")
                st.caption("Correlation between number of attempts and grade outcome, combined across all uploaded files.")
                if not attempt_frame.empty:
                    result = build_scatter_figure(attempt_frame, quiz_grade_type, colorblind_mode=colorblind_mode)
                    if result is not None:
                        fig, correlation, y_label, _ = result
                        fig.update_layout(template="plotly")
                        st.write(f"Correlation between Attempts and Quiz {y_label}: r = {correlation:.2f}")
                        st.plotly_chart(fig, use_container_width=True, key="quiz_scatter")
                else:
                    st.info("No quiz attempt data available yet.")

        if show_quiz_linegraph:
            with st.container(border=True):
                st.subheader("6. Line Graph of Various Metrics")
                st.caption("Trend of selected metrics across quizzes, combined across all uploaded files.")
                if not attempt_frame.empty:
                    if selected_quiz_metrics:
                        trend_data = build_metric_trend_data(attempt_frame, selected_quiz_metrics)
                        fig = build_line_graph_figure(trend_data, colorblind_mode=colorblind_mode)
                        fig.update_layout(template="plotly")
                        st.plotly_chart(fig, use_container_width=True, key="quiz_linegraph")
                else:
                    st.info("No quiz attempt data available yet.")

        render_pdf_report_panel(response_df, quiz_names, colorblind_mode)

else:
    # Pre-upload description & export guide
    with st.container(border=True):
        st.markdown("### 📈 Quiz Analysis")
        st.write("This section combines every uploaded Moodle STACK quiz file into one cohort-level view. Use the sidebar to upload one or more quiz responses files. After upload, you can:")
        st.markdown(
            """
            - review a merged list of every student attempt across every uploaded file
            - compare aggregated stats per quiz (participation, mean grade, variance)
            - inspect grade distributions per quiz as box plots
            - see engagement over time and how attempt counts relate to grades
            - track metric trends across quizzes
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
