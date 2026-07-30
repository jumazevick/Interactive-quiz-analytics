from __future__ import annotations

import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.parser import get_attempt_pools
from analytics.pdf_export import generate_pdf_report
from analytics.prt_transitions import (
    build_aggregate_graph,
    build_student_node_sequence,
    build_transition_graph_figure,
    build_transition_pairs,
    compute_network_features,
    count_question_parts,
)
from analytics.solution_distance import (
    build_prt_distance_3d_figure,
    build_ted_distance_3d_figure,
    compute_ted_distance_series,
)
from analytics.ui_theme import humanize_columns, inject_global_styles
from analytics.upload_ui import inject_sidebar_css, load_shared_response_df, render_options_panel

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
st.caption(
    "Visualizes how individual students — and the class as a whole — moved between "
    "PRT-classified answer types across quiz retakes, and how far each submission sat "
    "from the correct answer, measured two ways: by PRT-node depth, and by Tree Edit "
    "Distance between the submitted and correct CAS expressions."
)

uploaded_files, anonymize_data = render_options_panel()

quiz_metadata, response_df, quiz_names = load_shared_response_df(uploaded_files, anonymize_data)
selected_quiz_name = None

if quiz_names:
    # Provisional default; the real selector is rendered in the main pane below, next to
    # the question and part selectors. Keeping it out of the sidebar is what stops a long
    # list of uploaded quizzes from opening its dropdown near the bottom of the window
    # where BaseWeb pushes the lower entries off screen and out of reach.
    selected_quiz_name = quiz_names[0]


def _q_num(q_name: str) -> int:
    m = re.search(r"\d+", str(q_name))
    return int(m.group(0)) if m else 0


if not uploaded_files:
    with st.container(border=True):
        st.markdown("### 🧭 Solution Process Visualization")
        st.write(
            "Upload a Moodle **Responses** export (with the Question text / Response / "
            "Right answer display options enabled) via the sidebar. Files already "
            "uploaded on the Question & Quiz Analysis page are picked up here "
            "automatically."
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
elif response_df.empty:
    st.info("No usable question rows were found in the uploaded files.")
else:
    # Quiz / question / part are chosen together here in the main pane rather than in the
    # sidebar. `st.columns` hands back containers that can be written to in any order, so
    # each selector is filled in only once the previous one's value is known.
    quiz_col, question_col, part_col = st.columns(3)

    with quiz_col:
        selected_quiz_name = st.selectbox("Select Quiz", quiz_names, index=0)

    selected_df = response_df[response_df["quiz_name"] == selected_quiz_name].copy()
    pool_a_df, _ = get_attempt_pools(selected_df)

    question_order = sorted(pool_a_df["question"].unique(), key=_q_num)
    if not question_order:
        st.info("No questions found for this quiz.")
        st.stop()

    with question_col:
        selected_question = st.selectbox("Select Question", question_order, index=0)

    # A STACK question can be split into several parts, one PRT each (prt1, prt2, ...).
    # Each part is scored and classified independently, so every chart on this page is
    # scoped to one part at a time rather than silently reporting on part 1 alone.
    part_count = count_question_parts(pool_a_df, selected_question)
    with part_col:
        selected_part = st.selectbox(
            "Select Part",
            list(range(1, part_count + 1)),
            index=0,
            format_func=lambda p: f"Part {p} of {part_count}",
            help="This question's PRT parts (prt1, prt2, ...). Every graph below is scoped to the selected part.",
            disabled=part_count == 1,
        )
    if part_count == 1:
        st.caption("This question has a single part.")

    aggregate_fig = None
    network_features = pd.DataFrame()
    network_feature_figs: list[dict] = []

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader(f"1. Solution Process Transition Graphs — Part {selected_part}")
        st.caption(
            "**c** = full marks on this part; a numbered node is the classified wrong "
            "answer the part's PRT matched (PRT node 2 → **1**, node 3 → **2**, and so "
            "on); **0** = an unclassified wrong answer, meaning the response fell "
            "through every node without matching any of them, or was blank/invalid."
        )

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

        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"**Class-wide aggregate transition graph — {selected_question}, part {selected_part}**")
        st.caption("Edge thickness and color both scale with how many students made that transition (green = few, red = many).")
        agg_nodes, agg_edges = build_aggregate_graph(pool_a_df, selected_question, selected_part)
        if not agg_edges:
            st.info("Not enough multi-attempt data for this question to build an aggregate graph.")
        else:
            aggregate_fig = build_transition_graph_figure(
                agg_nodes, agg_edges, colorblind_mode=colorblind_mode,
                title=f"Class-wide Answer Transitions — {selected_question} (part {selected_part})",
            )
            st.plotly_chart(aggregate_fig, use_container_width=True, key="aggregate_graph")

            network_features = compute_network_features(agg_nodes, agg_edges)
            st.write("**Network features per node**")
            st.dataframe(humanize_columns(network_features), use_container_width=True, hide_index=True)

            feature_cols = st.columns(3)
            feature_specs = [
                ("in_degree_centrality", "In-Degree Centrality"),
                ("out_degree_centrality", "Out-Degree Centrality"),
                ("degree_centrality", "Degree Centrality"),
            ]
            # network_features["node"] is already in node_sort_key order ("0", "1", ...,
            # "c"); node values like "0"/"1"/"2" otherwise get silently coerced onto a
            # numeric axis by Plotly (dropping the non-numeric "c" bar entirely and
            # showing float tick marks in between), so the axis is forced categorical
            # in that exact order instead.
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
                network_feature_figs.append({"title": label, "figure": feature_fig})

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

        prt_fig = build_prt_distance_3d_figure(pool_a_df, selected_question, selected_part)
        st.plotly_chart(prt_fig, use_container_width=True, key="prt_distance_3d")

        ted_subset = compute_ted_distance_series(pool_a_df, selected_question, selected_part)
        unparsed = int(ted_subset["ted_distance"].isna().sum())
        if unparsed:
            st.caption(
                f"⚠️ {unparsed} response(s) for this part couldn't be parsed as a "
                "math expression and are excluded from the Tree Edit Distance chart below."
            )
        ted_fig = build_ted_distance_3d_figure(pool_a_df, selected_question, selected_part)
        st.plotly_chart(ted_fig, use_container_width=True, key="ted_distance_3d")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### 📄 PDF Report Options")
        st.caption(
            "Includes the class-wide aggregate graph, network feature charts, and both "
            "3D distance charts for the question and part selected above. The "
            "interactive single-student graph above is screen-only and isn't included."
        )

        part_label = f"{selected_question} part {selected_part}"
        pdf_sections = []
        if aggregate_fig is not None:
            pdf_sections.append({
                "title": f"Class-wide Answer Transitions — {part_label}",
                "caption": "Aggregated solution-process transition graph",
                "charts": [{"title": "Transition Graph", "figure": aggregate_fig}],
            })
            pdf_sections.append({
                "title": f"Network Features — {part_label}",
                "caption": "In-degree / out-degree / degree centrality per node",
                "df": humanize_columns(network_features),
                "charts": network_feature_figs,
            })
        pdf_sections.append({
            "title": f"3D Solution Process Distance — {part_label}",
            "caption": "PRT distance and Tree Edit Distance trajectories, colored by each point's own distance from the correct answer",
            "charts": [
                {"title": "PRT Distance", "figure": prt_fig},
                {"title": "Tree Edit Distance", "figure": ted_fig},
            ],
        })

        pdf_bytes = generate_pdf_report(
            title="Solution Process Visualization Report",
            subtitle=f"Quiz: {selected_quiz_name} • {part_label} • Generated Client-Side",
            sections=pdf_sections,
        )
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{selected_quiz_name}_{selected_question}_part{selected_part}_solution_process.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

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
