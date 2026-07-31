import pandas as pd

from analytics.validation import audit_question_data


def _row(attempt_idx, question, grade, response_status, overall_grade):
    return {
        "attempt_idx": attempt_idx, "question": question, "grade": grade, "max_grade": 1.0,
        "response_status": response_status, "response_text": "", "overall_grade": overall_grade,
        "student_name": "Jane Doe",
    }


def test_grade_mismatch_caused_by_an_ungraded_response_names_the_real_cause():
    # One question is "ungraded" (NaN, excluded from the mean) -- the calculated average
    # of the remaining questions can still land short of Moodle's own total for the
    # attempt (Moodle's total reflects the true, unknown-to-us score for that question),
    # and the warning should say so rather than blaming a "manual override".
    df = pd.DataFrame([
        _row(1, "Q1", float("nan"), "ungraded", 10.0),
        _row(1, "Q2", 1.0, "correct", 10.0),
        _row(1, "Q3", 0.0, "incorrect", 10.0),
    ])
    result = audit_question_data(df)
    joined = "\n".join(result["issues"])
    assert "ungraded" in joined.lower()
    assert result["checks"]["ungraded_count"] == 1


def test_grade_mismatch_without_any_ungraded_response_keeps_the_generic_message():
    df = pd.DataFrame([
        _row(1, "Q1", 0.0, "incorrect", 10.0),
        _row(1, "Q2", 1.0, "correct", 10.0),
    ])
    result = audit_question_data(df)
    joined = "\n".join(result["issues"])
    assert "manual grading overrides" in joined
    assert "ungraded" not in joined.lower()


def test_no_mismatch_when_calculated_and_moodle_grades_agree():
    df = pd.DataFrame([
        _row(1, "Q1", 1.0, "correct", 10.0),
        _row(1, "Q2", 1.0, "correct", 10.0),
    ])
    result = audit_question_data(df)
    assert result["is_valid"]
    assert not result["issues"]
