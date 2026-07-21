from __future__ import annotations

import re
from typing import Any
import pandas as pd

from analytics.parser import get_attempt_pools


def compute_question_metrics(response_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-question metrics using Pool A for participation and Pool B for performance."""
    if response_df.empty or "question" not in response_df.columns:
        return pd.DataFrame(columns=[
            "question", "attempts", "students", "invalid_rate", "blank_rate",
            "reattempt_share", "facility", "partial_credit_mean", "avg_score",
            "percent_correct", "percent_incorrect", "percent_valid", "percent_invalid",
            "syntax_error_count", "syntax_error_percent", "scaled_score", "catch_all_share"
        ])

    pool_a_df, pool_b_df = get_attempt_pools(response_df)

    def get_q_num(q_name: str) -> int:
        m = re.search(r"\d+", str(q_name))
        return int(m.group(0)) if m else 0

    questions = sorted(response_df["question"].unique(), key=get_q_num)
    rows = []

    for q in questions:
        q_a = pool_a_df[pool_a_df["question"] == q]
        q_b = pool_b_df[pool_b_df["question"] == q]

        # Pool A metrics (Participation / Usage)
        attempts_a = len(q_a)
        students_a = q_a["student_id"].nunique()
        invalid_count_a = int((q_a["response_status"] == "invalid").sum())
        blank_count_a = int((q_a["response_status"] == "blank").sum())
        invalid_rate_a = (invalid_count_a / attempts_a) if attempts_a > 0 else 0.0
        blank_rate_a = (blank_count_a / attempts_a) if attempts_a > 0 else 0.0
        percent_valid_a = max(0.0, (1.0 - invalid_rate_a - blank_rate_a) * 100.0)
        reattempt_share_a = max(0.0, ((attempts_a - students_a) / attempts_a) * 100.0) if attempts_a > 0 else 0.0

        # Pool B metrics (Performance / Mastery)
        num_students_b = len(q_b)
        correct_count_b = int((q_b["grade"] == 1.0).sum())
        facility_b = (correct_count_b / num_students_b) if num_students_b > 0 else 0.0
        partial_credit_mean_b = float(q_b["grade"].mean()) if num_students_b > 0 else 0.0
        avg_score_b = partial_credit_mean_b * 10.0
        scaled_score_b = partial_credit_mean_b * 10.0

        percent_correct_b = facility_b * 100.0
        percent_incorrect_b = (1.0 - facility_b) * 100.0

        # Catch-all share over wrong attempts in Pool B
        wrong_b = q_b[q_b["grade"] < 1.0]
        catch_all_count_b = 0
        total_wrong_prts_b = 0
        for _, r in wrong_b.iterrows():
            prt_list = r.get("prt_list") or []
            for prt in prt_list:
                fraction = prt.get("fraction")
                answer_note = prt.get("answer_note") or ""
                if fraction is not None and fraction < 1.0:
                    total_wrong_prts_b += 1
                    if re.match(r"^prt\d+-\d+-[TF]$", str(answer_note).strip()):
                        catch_all_count_b += 1

        catch_all_share_b = (catch_all_count_b / total_wrong_prts_b * 100.0) if total_wrong_prts_b > 0 else 0.0

        rows.append({
            "question": q,
            "attempts": attempts_a,
            "students": students_a,
            "invalid_rate": round(invalid_rate_a, 4),
            "blank_rate": round(blank_rate_a, 4),
            "reattempt_share": round(reattempt_share_a, 2),
            "facility": round(facility_b, 4),
            "partial_credit_mean": round(partial_credit_mean_b, 4),
            "avg_score": round(avg_score_b, 2),
            "percent_correct": round(percent_correct_b, 2),
            "percent_incorrect": round(percent_incorrect_b, 2),
            "percent_valid": round(percent_valid_a, 2),
            "percent_invalid": round(invalid_rate_a * 100.0, 2),
            "syntax_error_count": invalid_count_a,
            "syntax_error_percent": round(invalid_rate_a * 100.0, 2),
            "scaled_score": round(scaled_score_b, 2),
            "catch_all_share": round(catch_all_share_b, 2),
        })

    return pd.DataFrame(rows)


def compute_question_summary(response_df: pd.DataFrame, prt_frame: pd.DataFrame = None) -> dict[str, Any]:
    """Compute high-level question analytics summary."""
    if response_df.empty:
        return {
            "total_questions": 0,
            "student_count": 0,
            "average_score": 0.0,
            "average_valid_submission_rate": 0.0,
            "average_correct_rate": 0.0,
            "syntax_error_count": 0,
        }

    pool_a_df, pool_b_df = get_attempt_pools(response_df)
    qm = compute_question_metrics(response_df)

    total_questions = qm["question"].nunique() if not qm.empty else 0
    student_count = pool_b_df["student_id"].nunique()

    average_score = float(qm["avg_score"].mean()) if not qm.empty else 0.0
    average_valid_submission_rate = float(qm["percent_valid"].mean()) if not qm.empty else 0.0
    average_correct_rate = float(qm["percent_correct"].mean()) if not qm.empty else 0.0
    syntax_error_count = int((pool_a_df["response_status"] == "invalid").sum())

    return {
        "total_questions": total_questions,
        "student_count": student_count,
        "average_score": round(average_score, 2),
        "average_valid_submission_rate": round(average_valid_submission_rate, 2),
        "average_correct_rate": round(average_correct_rate, 2),
        "syntax_error_count": syntax_error_count,
    }


def compute_ranked_difficulty(question_metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank questions by average score (hardest first)."""
    if question_metrics.empty:
        return pd.DataFrame()
    return question_metrics.sort_values(by="avg_score", ascending=True).reset_index(drop=True)
