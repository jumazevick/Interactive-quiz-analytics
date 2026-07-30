from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.anonymize import anonymize_response_df
from analytics.data_loader import load_quiz_data
from analytics.pdf_export import generate_pdf_report
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
from analytics.upload_cache import clear_uploaded_files, get_uploader_key, sync_uploaded_files


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

# Sidebar overflow fix: with 13 section checkboxes plus a quiz selector, the sidebar
# can outgrow the viewport and hide the quiz dropdown below the fold without scrolling.
# Target every testid Streamlit has used for the sidebar's scroll container across
# versions (stSidebarContent / stSidebarUserContent in current releases, the older
# `> div:first-child` structure in earlier ones) so this doesn't silently stop working
# on a Streamlit upgrade.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        overflow-y: auto !important;
    }
    section[data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        overflow-y: auto !important;
        max-height: 100vh !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    with st.sidebar.expander("📎 Uploaded Files", expanded=False):
        for f in uploaded_files:
            st.write(f.name)

if st.sidebar.button("🗑️ Clear / Reset All Uploaded Files", use_container_width=True):
    clear_uploaded_files()
    st.cache_data.clear()
    st.rerun()

anonymize_data = st.sidebar.checkbox("🔒 Anonymize Student Data", value=True)


quiz_metadata: list[dict[str, object]] = []
response_df = pd.DataFrame()
quiz_names: list[str] = []
selected_quiz_name = None

if uploaded_files:
    quiz_metadata, response_df = load_quiz_data(uploaded_files)
    if anonymize_data:
        response_df = anonymize_response_df(response_df)
    if not response_df.empty:
        quiz_names = [item["quiz_name"] for item in quiz_metadata]

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

show_quiz_merged = st.sidebar.checkbox("8. Merged List of Users and Files", key="show_quiz_merged")
show_quiz_summary = st.sidebar.checkbox("9. Summary of Quiz Stats", key="show_quiz_summary")
selected_quiz_stats: list[str] = []
if show_quiz_summary:
    selected_quiz_stats = st.sidebar.multiselect(
        "Select Statistics to Display",
        ["student_count", "attempt_rate", "mean_grade", "grade_variance", "mean_highest_grade", "attempt_count"],
        default=["student_count", "attempt_rate", "mean_grade", "grade_variance", "mean_highest_grade", "attempt_count"],
        format_func=humanize_column_name,
    )
show_quiz_boxplot = st.sidebar.checkbox("10. Quiz Grade Distribution (Box Plot)", key="show_quiz_boxplot")
show_quiz_engagement = st.sidebar.checkbox("11. Engagement Over Time", key="show_quiz_engagement")
show_quiz_scatter = st.sidebar.checkbox("12. Scatter Plot: Attempts vs Grades", key="show_quiz_scatter")
quiz_grade_type = "Average Grade"
if show_quiz_scatter:
    quiz_grade_type = st.sidebar.radio("Select Grade Type", ("Highest Grade", "Average Grade", "Minimum Grade"))
show_quiz_linegraph = st.sidebar.checkbox("13. Line Graph of Various Metrics", key="show_quiz_linegraph")
selected_quiz_metrics: list[str] = []
if show_quiz_linegraph:
    selected_quiz_metrics = st.sidebar.multiselect(
        "Select Metrics to Display",
        ["student_count", "attempt_rate", "mean_grade", "grade_variance"],
        default=["student_count", "attempt_rate", "mean_grade", "grade_variance"],
        format_func=humanize_column_name,
    )

if uploaded_files:
    if response_df.empty:
        st.info("No usable question rows were found in the uploaded files.")
    else:
        st.caption(f"Combined across {len(quizzes_for_analysis)} of {len(quiz_names)} uploaded quiz file(s) — see the quiz selector in the sidebar.")

        # 8-13. Quiz Analysis (combined across the quizzes selected in the sidebar)
        attempt_frame = build_quiz_attempt_frame(response_df[response_df["quiz_name"].isin(quizzes_for_analysis)])

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
                quiz_merged_table = humanize_columns(attempt_frame)
                st.dataframe(quiz_merged_table, use_container_width=True, hide_index=True)

        if show_quiz_summary:
            with st.container(border=True):
                st.subheader("9. Summary of Quiz Stats")
                st.caption("Aggregated statistics per quiz, combined across all uploaded files.")
                if not attempt_frame.empty:
                    quiz_stats_df = compute_quiz_stats(attempt_frame, selected_quiz_stats)
                    quiz_summary_table = humanize_columns(quiz_stats_df)
                    st.dataframe(quiz_summary_table, use_container_width=True, hide_index=True)
                else:
                    st.info("No quiz attempt data available yet.")

        if show_quiz_boxplot:
            with st.container(border=True):
                st.subheader("10. Quiz Grade Distribution (Box Plot)")
                st.caption("Spread of grades per quiz, with mean grade overlay, combined across all uploaded files.")
                if not attempt_frame.empty:
                    fig = build_boxplot_figure(attempt_frame, colorblind_mode=colorblind_mode)
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
                    fig = build_engagement_figure(attempt_frame, colorblind_mode=colorblind_mode)
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
                    result = build_scatter_figure(attempt_frame, quiz_grade_type, colorblind_mode=colorblind_mode)
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
                    if selected_quiz_metrics:
                        trend_data = build_metric_trend_data(attempt_frame, selected_quiz_metrics)
                        fig = build_line_graph_figure(trend_data, colorblind_mode=colorblind_mode)
                        fig.update_layout(template="plotly")
                        st.plotly_chart(fig, use_container_width=True, key="quiz_linegraph")
                        quiz_linegraph_fig = fig
                else:
                    st.info("No quiz attempt data available yet.")

        # PDF Report Options — scoped to PDF generation only; the on-screen sections
        # above are unaffected by these controls. Only sections currently enabled
        # on-screen (via the sidebar checkboxes) are offered here, since their
        # underlying tables/charts are only computed when that toggle is on.
        quiz_section_options = [
            (show_quiz_merged, "8. Merged List of Users and Files"),
            (show_quiz_summary, "9. Summary of Quiz Stats"),
            (show_quiz_boxplot, "10. Quiz Grade Distribution (Box Plot)"),
            (show_quiz_engagement, "11. Engagement Over Time"),
            (show_quiz_scatter, "12. Scatter Plot: Attempts vs Grades"),
            (show_quiz_linegraph, "13. Line Graph of Various Metrics"),
        ]
        available_quiz_sections = [label for enabled, label in quiz_section_options if enabled]

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### 📄 PDF Report Options")
            pdf_selected_quiz_sections = available_quiz_sections
            if available_quiz_sections:
                pdf_selected_quiz_sections = st.multiselect(
                    "Select which Quiz Analysis sections to include",
                    options=available_quiz_sections,
                    default=available_quiz_sections,
                )

        pdf_sections = []
        if show_quiz_merged and quiz_merged_table is not None and "8. Merged List of Users and Files" in pdf_selected_quiz_sections:
            pdf_sections.append({"title": "8. Merged List of Users and Files", "caption": "All parsed quiz attempt rows (combined across uploaded files)", "df": quiz_merged_table})
        if show_quiz_summary and quiz_summary_table is not None and "9. Summary of Quiz Stats" in pdf_selected_quiz_sections:
            pdf_sections.append({"title": "9. Summary of Quiz Stats", "caption": "Aggregated stats per quiz", "df": quiz_summary_table})
        if show_quiz_boxplot and quiz_boxplot_fig is not None and "10. Quiz Grade Distribution (Box Plot)" in pdf_selected_quiz_sections:
            pdf_sections.append({"title": "10. Quiz Grade Distribution (Box Plot)", "caption": "Spread of grades per quiz, with mean grade overlay", "charts": [{"title": "Grade Distribution", "figure": quiz_boxplot_fig}]})
        if show_quiz_engagement and quiz_engagement_fig is not None and "11. Engagement Over Time" in pdf_selected_quiz_sections:
            pdf_sections.append({"title": "11. Engagement Over Time", "caption": "Density of quiz attempt start times per quiz", "charts": [{"title": "Engagement Over Time", "figure": quiz_engagement_fig}]})
        if show_quiz_scatter and quiz_scatter_fig is not None and "12. Scatter Plot: Attempts vs Grades" in pdf_selected_quiz_sections:
            pdf_sections.append({"title": "12. Scatter Plot: Attempts vs Grades", "caption": "Correlation between number of attempts and grade outcome", "charts": [{"title": "Attempts vs Grades", "figure": quiz_scatter_fig}]})
        if show_quiz_linegraph and quiz_linegraph_fig is not None and "13. Line Graph of Various Metrics" in pdf_selected_quiz_sections:
            pdf_sections.append({"title": "13. Line Graph of Various Metrics", "caption": "Trend of selected metrics across quizzes", "charts": [{"title": "Metrics by Quiz", "figure": quiz_linegraph_fig}]})

        pdf_bytes = generate_pdf_report(
            title="Moodle STACK Quiz Analysis Report",
            subtitle=f"{len(quizzes_for_analysis)} quiz file(s) combined • Generated Client-Side",
            sections=pdf_sections,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="quiz_analysis.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

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
