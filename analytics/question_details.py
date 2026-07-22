from __future__ import annotations

import pandas as pd

_NOT_AVAILABLE = "Not available in this export"


def build_question_detail(pool_b_df: pd.DataFrame, question: str) -> dict[str, str]:
    """Question text / right answer for a question, taken from Pool B (Best Attempt per
    Student). Falls back gracefully when the Moodle export didn't include these Display
    options — the columns always exist (parser.py fills them with "" otherwise)."""
    rows = pool_b_df[pool_b_df["question"] == question]

    def first_non_empty(column: str) -> str:
        if column not in rows.columns:
            return _NOT_AVAILABLE
        values = [v for v in rows[column].tolist() if isinstance(v, str) and v.strip()]
        return values[0] if values else _NOT_AVAILABLE

    return {
        "question_text": first_non_empty("question_text"),
        "right_answer_text": first_non_empty("right_answer_text"),
    }


def build_error_drilldown(pool_b_df: pd.DataFrame, question: str) -> pd.DataFrame:
    """Best-attempt rows for a question where the student didn't get full credit —
    lets a teacher scan submitted responses against the right answer to spot
    common misconceptions."""
    columns = ["student_name", "student_id", "response_text", "right_answer_text", "grade", "response_status"]
    wrong = pool_b_df[(pool_b_df["question"] == question) & (pool_b_df["grade"] < 1.0)].copy()
    if wrong.empty:
        return pd.DataFrame(columns=columns)

    # Defensive against a pool_b_df predating the question_text/right_answer_text
    # columns (e.g. a stale st.cache_data result computed before this feature existed).
    for column in columns:
        if column not in wrong.columns:
            wrong[column] = ""

    return (
        wrong[columns]
        .rename(columns={
            "student_name": "Student Name",
            "student_id": "Email",
            "response_text": "Submitted Response",
            "right_answer_text": "Right Answer",
            "grade": "Score",
            "response_status": "Status",
        })
        .reset_index(drop=True)
    )
