from __future__ import annotations

from typing import Any

import pandas as pd


def audit_question_data(response_df: pd.DataFrame) -> dict[str, Any]:
    """Validate that the parsed response data contains the fields needed for the dashboard."""
    total_attempts = int(response_df["attempt_idx"].nunique()) if "attempt_idx" in response_df.columns else 0
    question_count = int(response_df["question"].nunique()) if "question" in response_df.columns else 0

    checks: dict[str, Any] = {
        "row_count": total_attempts,
        "question_count": question_count,
        "has_question_column": "question" in response_df.columns,
        "has_grade_column": "grade" in response_df.columns,
        "has_max_grade_column": "max_grade" in response_df.columns,
        "has_response_status_column": "response_status" in response_df.columns,
        "has_response_text_column": "response_text" in response_df.columns,
    }

    issues: list[str] = []
    if total_attempts == 0:
        issues.append("No response rows were parsed from the uploaded export.")
    if not checks["has_question_column"]:
        issues.append("The uploaded data is missing question labels.")
    if not checks["has_grade_column"] or not checks["has_max_grade_column"]:
        issues.append("Question scores and maxima could not be resolved from the export.")
    if not checks["has_response_status_column"]:
        issues.append("Response status values are missing, so outcome percentages cannot be calculated.")

    if total_attempts > 0 and checks["has_response_status_column"]:
        # Tally counts across all questions in the quiz
        checks["syntax_error_count"] = int((response_df["response_status"] == "invalid").sum())
        checks["invalid_count"] = int((response_df["response_status"] == "invalid").sum())
        checks["blank_count"] = int((response_df["response_status"] == "blank").sum())

    # Automated Grade Verification / Cross-check (Part 5)
    mismatches = []
    if not response_df.empty and "attempt_idx" in response_df.columns and "grade" in response_df.columns and "overall_grade" in response_df.columns:
        for attempt_id, group in response_df.groupby("attempt_idx"):
            calculated_grade = 10.0 * group["grade"].mean()
            actual_grade = float(group["overall_grade"].iloc[0])
            if abs(calculated_grade - actual_grade) >= 0.01:
                student_name = group["student_name"].iloc[0]
                mismatches.append(f"Student: {student_name} (Row {attempt_id}) - Calculated={calculated_grade:.2f}, Moodle={actual_grade:.2f}")

    if mismatches:
        import sys
        print(f"Validation warning: Grade mismatch detected in {len(mismatches)} rows:", file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        
        issues.append("Grade validation notice: Mismatches between calculated question-average scores and Moodle's overall attempt grades were found (likely due to manual grading overrides or regrades in Moodle):")
        for m in mismatches[:10]:  # Show first 10 to avoid UI clutter
            issues.append(f"  • {m}")
        if len(mismatches) > 10:
            issues.append(f"  • ... and {len(mismatches) - 10} more rows.")

    return {"checks": checks, "issues": issues, "is_valid": len(issues) == 0}
