from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from analytics.parser import (
    build_grade_breakdown_rows,
    build_response_rows,
    detect_export_type,
    merge_grade_breakdown_rows,
    parse_uploaded_file,
)
from analytics.upload_cache import CACHE_HASH_FUNCS

_EMPTY_RESPONSE_COLUMNS = [
    "student_id", "student_name", "question", "grade", "max_grade",
    "response_status", "response_text", "quiz_name", "overall_grade",
    "completed_dt", "started_on", "attempt_idx", "source_type",
    "question_text", "right_answer_text",
]


@st.cache_data(show_spinner=False, hash_funcs=CACHE_HASH_FUNCS)
def load_quiz_data(files) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Parse every uploaded Moodle export into one combined, long-format response
    DataFrame plus per-file quiz metadata. Shared by every page (via the shared upload
    cache in `analytics.upload_cache`) so an upload made on one page is immediately
    usable on another without re-parsing."""
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
            parsed_df = pd.DataFrame(columns=_EMPTY_RESPONSE_COLUMNS)

        if not parsed_df.empty:
            parsed_df["quiz_id"] = index
            parsed_df["quiz_name"] = quiz_name
            quiz_groups.setdefault(quiz_name, []).append(parsed_df)
        quiz_metadata.append({"quiz_id": index, "quiz_name": quiz_name})

    if not quiz_groups:
        return quiz_metadata, pd.DataFrame(columns=_EMPTY_RESPONSE_COLUMNS + ["quiz_id"])

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
