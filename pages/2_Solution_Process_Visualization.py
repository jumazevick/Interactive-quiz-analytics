from __future__ import annotations

import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.parser import get_attempt_pools
from analytics.pdf_ui import render_pdf_report_panel
from analytics.prt_transitions import (
    build_aggregate_graph,
    build_student_node_sequence,
    build_transition_graph_figure,
    build_transition_pairs,
    compute_network_features,
    count_question_parts,
)
from analytics.solution_distance import (
    CROSS_ATTEMPT_METRICS,
    build_cross_attempt_figure,
    build_prt_distance_3d_figure,
    build_ted_distance_3d_figure,
    classify_cross_attempt_trends,
    compute_cross_attempt_comparison,
    compute_ted_distance_series,
)
from analytics.ui_theme import humanize_columns, inject_global_styles
from analytics.upload_ui import inject_sidebar_css, load_shared_response_df, render_options_panel, render_sidebar_bottom_spacer

st.set_page_config(
    page_title="Solution Process Visualization",
    page_icon=":compass:",
    layout="wide",
)
inject_global_styles()

inject_sidebar_css()

title_col, colorblind_col = st.columns([5, 2])
with title_col:
    st.title("Solution Process Visualization")
with colorblind_col:
    st.markdown("<div style='margin-top: 1.6rem;'></div>", unsafe_allow_html=True)
    colorblind_mode = st.toggle(
        "🎨 Colorblind Mode",
        key="solution_process_colorblind_mode",
        help="Switches every chart on this page to a colorblind-safe palette.",
    )
st.warning("⏳ Depending on the size of your upload, it may take up to 30 seconds for all statistics to fully render, and up to 30 seconds for the downloadable PDF report to generate.")

uploaded_files, anonymize_data = render_options_panel()

quiz_metadata, response_df, quiz_names = load_shared_response_df(uploaded_files, anonymize_data)


def _q_num(q_name: str) -> int:
    m = re.search(r"\d+", str(q_name))
    return int(m.group(0)) if m else 0


# --- Sidebar: Solution Process Visualization scope, then its section toggles — the same
# layout the Question Analysis and Quiz Analysis sidebars use. The three scope selectors
# are chained (a quiz decides which questions exist, a question decides how many parts),
# so each one is only rendered once the previous one's value is known. ---
st.sidebar.markdown("---")
st.sidebar.subheader("🧭 Solution Process Visualization")
st.sidebar.caption("Applies to the single quiz, question and part selected below")

selected_quiz_name = None
selected_question = None
selected_part = 1
part_count = 1
pool_a_df = pd.DataFrame()
question_order: list[str] = []

if quiz_names and not response_df.empty:
    selected_quiz_name = st.sidebar.selectbox("Select Quiz", quiz_names, index=0)
    selected_df = response_df[response_df["quiz_name"] == selected_quiz_name].copy()
    pool_a_df, _ = get_attempt_pools(selected_df)
    question_order = sorted(pool_a_df["question"].unique(), key=_q_num)

    if question_order:
        selected_question = st.sidebar.selectbox("Select Question", question_order, index=0)
        # A STACK question can be split into several parts, one PRT each (prt1, prt2, ...).
        # Each part is scored and classified independently, so every chart on this page is
        # scoped to one part at a time rather than silently reporting on part 1 alone.
        part_count = count_question_parts(pool_a_df, selected_question)
        selected_part = st.sidebar.selectbox(
            "Select Part",
            list(range(1, part_count + 1)),
            index=0,
            format_func=lambda p: f"Part {p} of {part_count}",
            help="This question's PRT parts (prt1, prt2, ...). Every graph on the page is scoped to the selected part.",
            disabled=part_count == 1,
        )

_SPV_SECTION_KEYS = [
    "show_spv_student_graph", "show_spv_aggregate_graph",
    "show_spv_network_features", "show_spv_prt_3d", "show_spv_ted_3d",
    "show_spv_cross_attempt",
]
spv_select_col, spv_deselect_col = st.sidebar.columns(2)
# Setting session_state here, before the checkboxes below are instantiated in this same
# script run, is what makes the new value take effect immediately — a checkbox's `value=`
# argument is only its default the very first time a key is seen.
if spv_select_col.button("Select All", key="spv_select_all", use_container_width=True):
    for _key in _SPV_SECTION_KEYS:
        st.session_state[_key] = True
if spv_deselect_col.button("Deselect All", key="spv_deselect_all", use_container_width=True):
    for _key in _SPV_SECTION_KEYS:
        st.session_state[_key] = False

show_student_graph = st.sidebar.checkbox("1. Single-Student Transition Graph", value=True, key="show_spv_student_graph")
show_aggregate_graph = st.sidebar.checkbox("2. Class-Wide Transition Graph", value=True, key="show_spv_aggregate_graph")
show_network_features = st.sidebar.checkbox("3. Network Features per Node", value=True, key="show_spv_network_features")
show_prt_3d = st.sidebar.checkbox("4. PRT-Distance 3D Chart", value=True, key="show_spv_prt_3d")
show_ted_3d = st.sidebar.checkbox("5. Tree Edit Distance 3D Chart", value=True, key="show_spv_ted_3d")
show_cross_attempt = st.sidebar.checkbox("6. Cross-Attempt Comparison", value=True, key="show_spv_cross_attempt")
cross_attempt_metric = "Grade"
if show_cross_attempt:
    cross_attempt_metric = st.sidebar.radio(
        "Compare by",
        list(CROSS_ATTEMPT_METRICS.keys()),
        key="cross_attempt_metric",
        help="Grade covers the whole question (every part combined) and ignores the Part selector above; the two distance metrics are scoped to the selected part, same as the 3D charts.",
    )

render_sidebar_bottom_spacer()

if not uploaded_files:
    with st.container(border=True):
        st.markdown("### 🧭 Solution Process Visualization")
        st.write(
            "This section visualizes how individual students — and the class as a whole — "
            "moved between PRT-classified answer types across quiz retakes, and how far "
            "each submission sat from the correct answer, measured two ways: by PRT-node "
            "depth, and by Tree Edit Distance between the submitted and correct CAS "
            "expressions. Use the sidebar to upload one or more quiz responses files, with "
            "the Question text / Response / Right answer display options enabled. After "
            "upload, you can:"
        )
        st.markdown(
            """
            - see how one student's answers moved between PRT-classified answer types
              (unclassified wrong, a specific classified wrong answer, correct) across
              their quiz retakes, as a directed graph
            - see the whole class's aggregate transition graph for a question, with
              edge thickness/color showing how many students made each transition
            - review in/out-degree and degree centrality for each answer-type node
            - see each student's trajectory toward the correct answer in 3D, measured
              by PRT-node distance and by Tree Edit Distance between their submitted
              and correct math expressions
            """
        )
        st.info(
            "This page needs multiple quiz **retakes** per student (several rows for "
            "the same student in the export) to show a meaningful trajectory — a "
            "single-attempt export will only show single-point trajectories."
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
elif response_df.empty:
    st.info("No usable question rows were found in the uploaded files.")
elif not question_order:
    st.info("No questions found for this quiz.")
else:
    st.caption(
        f"Showing **{selected_quiz_name}** — {selected_question}, "
        + (f"part {selected_part} of {part_count}" if part_count > 1 else "single part")
        + ". Change the quiz, question or part in the sidebar."
    )

    if show_student_graph or show_aggregate_graph or show_network_features:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader(f"1. Solution Process Transition Graphs — Part {selected_part}")
            st.caption(
                "**c** = full marks on this part; a numbered node is the classified wrong "
                "answer the part's PRT matched (PRT node 2 → **1**, node 3 → **2**, and so "
                "on); **0** = an unclassified wrong answer, meaning the response fell "
                "through every node without matching any of them, or was blank/invalid."
            )

            if show_student_graph:
                question_rows = pool_a_df[pool_a_df["question"] == selected_question]
                grade_lookup = question_rows.groupby("student_id")["overall_grade"].max().reset_index()
                attempts_lookup = question_rows.groupby("student_id").size().rename("attempts").reset_index()

                roster = (
                    question_rows[["student_id", "student_name"]]
                    .drop_duplicates()
                    .merge(grade_lookup, on="student_id", how="left")
                    .merge(attempts_lookup, on="student_id", how="left")
                    .sort_values(by="student_name")
                    .reset_index(drop=True)
                )
                roster["overall_grade"] = roster["overall_grade"].round(2)
                roster_display = roster.rename(columns={
                    "student_name": "Student Name",
                    "overall_grade": "Overall Grade",
                    "attempts": "Attempts on this Question",
                })[["Student Name", "Overall Grade", "Attempts on this Question"]]

                st.write("**Select a student** to see their individual solution process:")
                roster_event = st.dataframe(
                    roster_display,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="solution_process_roster",
                )

                selected_rows = roster_event.selection.rows if roster_event and roster_event.selection else []
                if selected_rows:
                    selected_row = roster.iloc[selected_rows[0]]
                    selected_student_id = selected_row["student_id"]
                    selected_student_name = selected_row["student_name"]
                    seq = build_student_node_sequence(pool_a_df, selected_question, selected_student_id, selected_part)
                    st.write(f"**{selected_student_name}** — {len(seq)} attempt(s) on {selected_question}, part {selected_part}")
                    if len(seq) < 2:
                        st.info("This student has only one recorded attempt at this question — no transition to show.")
                    else:
                        student_nodes = seq["node"].tolist()
                        student_edges = Counter(build_transition_pairs(student_nodes))
                        student_fig = build_transition_graph_figure(
                            student_nodes,
                            student_edges,
                            colorblind_mode=colorblind_mode,
                            title=f"{selected_student_name}'s Answer Transitions — {selected_question} (part {selected_part})",
                        )
                        st.plotly_chart(student_fig, use_container_width=True, key="single_student_graph")
                else:
                    st.caption("Click a row above to see that student's individual transition graph.")

            # Built whenever either of the two blocks below is on: the network features are
            # computed from this same graph, so they can be shown with the graph itself
            # hidden.
            agg_nodes, agg_edges = ([], {})
            if show_aggregate_graph or show_network_features:
                agg_nodes, agg_edges = build_aggregate_graph(pool_a_df, selected_question, selected_part)

            if show_aggregate_graph:
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Class-wide aggregate transition graph — {selected_question}, part {selected_part}**")
                st.caption("Edge thickness and color both scale with how many students made that transition (green = few, red = many).")
                if not agg_edges:
                    st.info("Not enough multi-attempt data for this question to build an aggregate graph.")
                else:
                    st.plotly_chart(
                        build_transition_graph_figure(
                            agg_nodes, agg_edges, colorblind_mode=colorblind_mode,
                            title=f"Class-wide Answer Transitions — {selected_question} (part {selected_part})",
                        ),
                        use_container_width=True, key="aggregate_graph",
                    )

            if show_network_features:
                if not agg_edges:
                    st.info("Not enough multi-attempt data for this question to compute network features.")
                else:
                    network_features = compute_network_features(agg_nodes, agg_edges)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.write("**Network features per node**")
                    st.dataframe(humanize_columns(network_features), use_container_width=True, hide_index=True)

                    feature_cols = st.columns(3)
                    feature_specs = [
                        ("in_degree_centrality", "In-Degree Centrality"),
                        ("out_degree_centrality", "Out-Degree Centrality"),
                        ("degree_centrality", "Degree Centrality"),
                    ]
                    # network_features["node"] is already in node_sort_key order ("0", "1",
                    # ..., "c"); node values like "0"/"1"/"2" otherwise get silently coerced
                    # onto a numeric axis by Plotly (dropping the non-numeric "c" bar
                    # entirely and showing float tick marks in between), so the axis is
                    # forced categorical in that exact order instead.
                    node_order = network_features["node"].tolist()
                    for col, (metric, label) in zip(feature_cols, feature_specs):
                        feature_fig = px.bar(
                            network_features, x="node", y=metric,
                            category_orders={"node": node_order},
                            labels={"node": "Node", metric: label},
                        )
                        feature_fig.update_traces(marker_color="#3b82f6")
                        feature_fig.update_xaxes(type="category")
                        feature_fig.update_layout(title=label, showlegend=False)
                        col.plotly_chart(feature_fig, use_container_width=True, key=f"network_feature_{metric}")

    if show_prt_3d or show_ted_3d:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader(f"2. 3D Solution Process Distance Visualizations — Part {selected_part}")
            st.caption(
                "Each line is one student's trajectory across their attempts at this part. "
                "Every point is colored by its own distance from the correct answer — white "
                "at 0, neon red at 1, then running up through orange, yellow, green, and "
                "blue to black at the largest distance seen. Students are ordered along the "
                "**Students** axis by their first attempt's distance, then by their second "
                "attempt within each of those groups, and so on — so a student correct on "
                "the first try sits closest to the origin as a white dot. The PRT distance "
                "is a heuristic generalization of a teacher-authored distance table (Takada "
                "et al., EDM 2025) — treat it as relative, not an absolute or calibrated "
                "measure."
            )

            if show_prt_3d:
                st.plotly_chart(
                    build_prt_distance_3d_figure(pool_a_df, selected_question, selected_part),
                    use_container_width=True, key="prt_distance_3d",
                )

            if show_ted_3d:
                ted_subset = compute_ted_distance_series(pool_a_df, selected_question, selected_part)
                unparsed = int(ted_subset["ted_distance"].isna().sum())
                if unparsed:
                    st.caption(
                        f"⚠️ {unparsed} response(s) for this part couldn't be parsed as a "
                        "math expression and are excluded from the Tree Edit Distance chart below."
                    )
                st.plotly_chart(
                    build_ted_distance_3d_figure(pool_a_df, selected_question, selected_part),
                    use_container_width=True, key="ted_distance_3d",
                )

    if show_cross_attempt:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader(f"3. Cross-Attempt Comparison — {cross_attempt_metric}")
            st.caption(
                "For every student who retook this question, how did their "
                f"{cross_attempt_metric.lower()} change from their first attempt to their "
                "last? Only students with 2 or more attempts are shown — a single attempt "
                "has no change to compare."
            )

            higher_is_better = bool(CROSS_ATTEMPT_METRICS[cross_attempt_metric]["higher_is_better"])
            comparison = compute_cross_attempt_comparison(
                pool_a_df, selected_question, cross_attempt_metric, selected_part,
            )
            trends = classify_cross_attempt_trends(comparison, higher_is_better)

            if comparison.empty:
                st.info(
                    f"No students have 2 or more attempts with a usable {cross_attempt_metric.lower()} "
                    f"value on {selected_question}"
                    + (f", part {selected_part}" if cross_attempt_metric != "Grade" else "")
                    + " — nothing to compare across attempts yet."
                )
            else:
                trend_counts = trends["trend"].value_counts()
                total_students = len(trends)
                summary_cols = st.columns(3)
                for col, trend, emoji in zip(summary_cols, ["Improved", "Flat", "Regressed"], ["📈", "➖", "📉"]):
                    count = int(trend_counts.get(trend, 0))
                    percent = 100 * count / total_students if total_students else 0.0
                    col.metric(f"{emoji} {trend}", f"{count} student(s)", f"{percent:.0f}% of {total_students}")

                st.plotly_chart(
                    build_cross_attempt_figure(comparison, trends, cross_attempt_metric, colorblind_mode),
                    use_container_width=True, key="cross_attempt_chart",
                )

                st.write("**Ranked by change, most improved first** (positive = improved, regardless of whether this metric counts up or down when things get better):")
                ranking_table = trends.rename(columns={
                    "student_name": "Student Name",
                    "first_value": "First Attempt",
                    "last_value": "Last Attempt",
                    "change": "Change",
                    "trend": "Trend",
                })[["Student Name", "First Attempt", "Last Attempt", "Change", "Trend"]]
                st.dataframe(humanize_columns(ranking_table), use_container_width=True, hide_index=True)

    render_pdf_report_panel(response_df, quiz_names, colorblind_mode)

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
