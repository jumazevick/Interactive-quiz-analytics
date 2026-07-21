from __future__ import annotations

from typing import Any

import pandas as pd


def audit_question_data(response_df: pd.DataFrame) -> dict[str, Any]:
    """Validate that the parsed response data contains the fields needed for the dashboard."""
    checks: dict[str, Any] = {
        "row_count": int(len(response_df)),
        "question_count": int(response_df["question"].nunique()) if "question" in response_df.columns else 0,
        "has_question_column": "question" in response_df.columns,
        "has_grade_column": "grade" in response_df.columns,
        "has_max_grade_column": "max_grade" in response_df.columns,
        "has_response_status_column": "response_status" in response_df.columns,
        "has_response_text_column": "response_text" in response_df.columns,
    }

    issues: list[str] = []
    if checks["row_count"] == 0:
        issues.append("No response rows were parsed from the uploaded export.")
    if not checks["has_question_column"]:
        issues.append("The uploaded data is missing question labels.")
    if not checks["has_grade_column"] or not checks["has_max_grade_column"]:
        issues.append("Question scores and maxima could not be resolved from the export.")
    if not checks["has_response_status_column"]:
        issues.append("Response status values are missing, so outcome percentages cannot be calculated.")

    if checks["row_count"] > 0 and checks["has_response_status_column"]:
        status_counts = response_df["response_status"].value_counts(dropna=False)
        if "syntax_error" in status_counts.index:
            checks["syntax_error_count"] = int(status_counts["syntax_error"])
        if "invalid" in status_counts.index:
            checks["invalid_count"] = int(status_counts["invalid"])
        if "blank" in status_counts.index:
            checks["blank_count"] = int(status_counts["blank"])

    return {"checks": checks, "issues": issues, "is_valid": len(issues) == 0}
