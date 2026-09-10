from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.parser import get_attempt_pools
from analytics.pdf_export import generate_pdf_report
from analytics.prt_transitions import count_question_parts
from analytics.report_sections import (
    QUESTION_MODULES,
    QUIZ_MODULES,
    SPV_MODULES,
    _q_num,
    build_report_sections,
)

# The rendered report and the selections it was built from. Plain session keys, not widget
# keys: Streamlit garbage-collects widget state for widgets that weren't rendered on the
# previous run (a page switch counts), and a generated report should outlive that.
_PDF_BYTES_KEY = "pdf_report_bytes"
_PDF_NAME_KEY = "pdf_report_filename"
_PDF_SIGNATURE_KEY = "pdf_report_signature"


def _spv_scope_options(response_df: pd.DataFrame, selected_quizzes: list[str]) -> tuple[list[str], dict[str, int]]:
    """Questions available across the selected quizzes, and the part count for each."""
    scoped = response_df[response_df["quiz_name"].isin(selected_quizzes)]
    if scoped.empty:
        return [], {}

    pool_a_df, _ = get_attempt_pools(scoped)
    if pool_a_df.empty:
        return [], {}

    questions = sorted(pool_a_df["question"].unique(), key=_q_num)
    return questions, {q: count_question_parts(pool_a_df, q) for q in questions}


def render_pdf_report_panel(
    response_df: pd.DataFrame,
    quiz_names: list[str],
    colorblind_mode: bool,
) -> None:
    """The shared PDF report builder, rendered at the foot of all three analysis pages.

    Deliberately page-independent: the controls, the defaults and the resulting PDF are the
    same wherever it is rendered, so the page being viewed never pre-filters the report.
    That is also why the section builders in `analytics.report_sections` construct their own
    figures instead of reusing whatever the current page happened to draw.
    """
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### 📄 PDF Report Options")
        st.caption(
            "The same options on every analysis page — whichever page you are on, the same "
            "selections produce the same report. Sections are concatenated in order: "
            "Question Analysis (each quiz's full run in turn), then Solution Process "
            "Visualization, then Quiz Analysis."
        )

        include_question = st.checkbox("Include Question Analysis", value=True, key="pdf_include_question")
        question_sections: list[str] | None = None
        if include_question:
            question_sections = st.multiselect(
                "Select which Question Analysis sections to include",
                options=QUESTION_MODULES,
                default=QUESTION_MODULES,
                key="pdf_question_sections",
            )

        include_spv = st.checkbox("Include Solution Process Visualization", value=True, key="pdf_include_spv")
        spv_sections: list[str] | None = None
        spv_question: str | None = None
        spv_part = 1
        if include_spv:
            spv_sections = st.multiselect(
                "Select which Solution Process Visualization sections to include",
                options=SPV_MODULES,
                default=SPV_MODULES,
                key="pdf_spv_sections",
            )

        include_quiz = st.checkbox("Include Quiz Analysis", value=True, key="pdf_include_quiz")
        quiz_sections: list[str] | None = None
        if include_quiz:
            quiz_sections = st.multiselect(
                "Select which Quiz Analysis sections to include",
                options=QUIZ_MODULES,
                default=QUIZ_MODULES,
                key="pdf_quiz_sections",
            )

        selected_quizzes = st.multiselect(
            "Select which quiz(zes) to include in the PDF report",
            options=quiz_names,
            default=quiz_names,
            key="pdf_quizzes",
        )

        # The Solution Process sections are scoped to one question and part, so the panel
        # carries its own selector rather than borrowing the SPV page's — that keeps the
        # report identical from every page. Rendered after the quiz multiselect because its
        # options depend on which quizzes are in.
        if include_spv:
            questions, part_counts = _spv_scope_options(response_df, selected_quizzes)
            if not questions:
                st.caption("⚠️ No questions found in the selected quizzes for the Solution Process sections.")
                spv_sections = None
            else:
                scope_cols = st.columns(2)
                spv_question = scope_cols[0].selectbox(
                    "Solution Process: question",
                    options=questions,
                    key="pdf_spv_question",
                    help="The Solution Process charts cover one question and part at a time; this picks which, for every selected quiz.",
                )
                part_count = part_counts.get(spv_question, 1)
                spv_part = scope_cols[1].selectbox(
                    "Solution Process: part",
                    options=list(range(1, part_count + 1)),
                    format_func=lambda p: f"Part {p} of {part_count}",
                    key="pdf_spv_part",
                    disabled=part_count == 1,
                )

        signature = (
            tuple(selected_quizzes),
            tuple(question_sections or ()),
            tuple(spv_sections or ()),
            tuple(quiz_sections or ()),
            spv_question,
            spv_part,
            colorblind_mode,
        )
        has_content = bool(selected_quizzes) and any((question_sections, spv_sections, quiz_sections))

        if not has_content:
            st.button(
                "📄 Download PDF Report",
                use_container_width=True,
                disabled=True,
                key="pdf_download_disabled",
            )
            st.caption("Tick at least one section and one quiz to enable the report.")
            return

        # Generation is a separate click rather than happening on every rerun: with three
        # sections across several quizzes there can be dozens of charts to rasterize, and
        # doing that on each widget interaction would make the page unusable.
        if st.button("⚙️ Generate PDF Report", use_container_width=True, key="pdf_generate"):
            with st.spinner("Building the report — rasterizing charts can take a while for several quizzes…"):
                sections = build_report_sections(
                    response_df=response_df,
                    selected_quizzes=selected_quizzes,
                    colorblind_mode=colorblind_mode,
                    question_sections=question_sections,
                    spv_sections=spv_sections,
                    quiz_sections=quiz_sections,
                    spv_question=spv_question,
                    spv_part=spv_part,
                )
                if not sections:
                    st.session_state.pop(_PDF_BYTES_KEY, None)
                    st.warning("Nothing to report: the selected quizzes have no data for the selected sections.")
                else:
                    st.session_state[_PDF_BYTES_KEY] = generate_pdf_report(
                        title="Moodle STACK Quiz Analytics Report",
                        subtitle=f"{len(selected_quizzes)} quiz file(s) • {len(sections)} section(s) • Generated Client-Side",
                        sections=sections,
                    )
                    st.session_state[_PDF_NAME_KEY] = (
                        f"{selected_quizzes[0]}_analytics_report.pdf" if len(selected_quizzes) == 1
                        else f"{len(selected_quizzes)}_quizzes_analytics_report.pdf"
                    )
                    st.session_state[_PDF_SIGNATURE_KEY] = signature

        pdf_bytes = st.session_state.get(_PDF_BYTES_KEY)
        if not pdf_bytes:
            return

        if st.session_state.get(_PDF_SIGNATURE_KEY) != signature:
            st.caption("⚠️ These options changed since the report was generated — generate again for an up-to-date PDF.")
            return

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=st.session_state.get(_PDF_NAME_KEY, "analytics_report.pdf"),
            mime="application/pdf",
            use_container_width=True,
            key="pdf_download",
        )
