from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def compute_question_metrics(response_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-question metrics used by the dashboard."""
    if response_df.empty:
        return pd.DataFrame(
            columns=[
                "question",
                "attempts",
                "students",
                "invalid_rate",
                "blank_rate",
                "reattempt_share",
                "facility",
                "partial_credit_mean",
                "avg_score",
                "percent_correct",
                "percent_incorrect",
                "percent_valid",
                "percent_invalid",
                "syntax_error_count",
                "syntax_error_percent",
                "scaled_score",
                "catch_all_share",
            ]
        )

    # Precompute reattempt share
    student_counts = response_df.groupby("student_id")["attempt_idx"].nunique()
    extra_attempts = sum(c - 1 for c in student_counts if c > 1)
    total_attempts_overall = response_df["attempt_idx"].nunique()
    reattempt_share_val = round(float(extra_attempts / total_attempts_overall * 100), 2) if total_attempts_overall > 0 else 0.0

    grouped = response_df.groupby("question")
    metrics = []
    for question, rows in grouped:
        attempts = len(rows)
        students = int(rows["student_id"].nunique())

        correct_count = int((rows["response_status"] == "correct").sum())
        incorrect_count = int((rows["response_status"] == "incorrect").sum())
        invalid_count = int((rows["response_status"] == "invalid").sum())
        blank_count = int((rows["response_status"] == "blank").sum())

        facility = correct_count / attempts if attempts else 0.0
        invalid_rate = invalid_count / attempts if attempts else 0.0
        blank_rate = blank_count / attempts if attempts else 0.0
        partial_credit_mean = float(rows["grade"].mean()) if attempts else 0.0

        percent_correct = round(facility * 100, 2)
        percent_incorrect = round((1 - facility) * 100, 2)
        percent_valid = round((1 - invalid_rate - blank_rate) * 100, 2)
        percent_invalid = round(invalid_rate * 100, 2)
        syntax_error_count = invalid_count
        syntax_error_percent = percent_invalid

        # M and catch-all share
        M = max([prt["index"] for prt_list in rows["prt_list"] for prt in prt_list] + [1])
        wrong_rows = rows[rows["grade"] < 1.0]
        total_wrong_notes = 0
        catch_all_notes = 0
        for _, row in wrong_rows.iterrows():
            prt_list = row.get("prt_list", [])
            prt_map = {prt["index"]: prt for prt in prt_list}
            for k in range(1, M + 1):
                note = "(invalid/blank input)"
                if k in prt_map:
                    note = prt_map[k]["answer_note"]

                total_wrong_notes += 1
                if re.match(r"^prt\d+-\d+-[TF]$", note):
                    catch_all_notes += 1

        catch_all_share = round(float(catch_all_notes / total_wrong_notes * 100), 2) if total_wrong_notes > 0 else 0.0

        metrics.append(
            {
                "question": question,
                "attempts": attempts,
                "students": students,
                "invalid_rate": round(invalid_rate, 4),
                "blank_rate": round(blank_rate, 4),
                "reattempt_share": reattempt_share_val,
                "facility": round(facility, 4),
                "partial_credit_mean": round(partial_credit_mean, 4),
                "avg_score": round(facility * 10.0, 2),  # avg_score displays facility * 10
                "percent_correct": percent_correct,
                "percent_incorrect": percent_incorrect,
                "percent_valid": percent_valid,
                "percent_invalid": percent_invalid,
                "syntax_error_count": syntax_error_count,
                "syntax_error_percent": syntax_error_percent,
                "scaled_score": round(partial_credit_mean * 10.0, 2),
                "catch_all_share": catch_all_share,
            }
        )

    metrics_df = pd.DataFrame(metrics)
    if metrics_df.empty:
        return metrics_df

    metrics_df = metrics_df.sort_values("question").reset_index(drop=True)
    return metrics_df


def compute_question_summary(response_df: pd.DataFrame, prt_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compute the overview summary cards for the page."""
    if response_df.empty:
        return {
            "total_questions": 0,
            "overall_prt_elements": 0,
            "most_difficult_question": "-",
            "syntax_error_count": 0,
            "average_score": 0.0,
            "average_valid_submission_rate": 0.0,
            "average_correct_rate": 0.0,
            "student_count": 0,
        }

    question_metrics = compute_question_metrics(response_df)
    if question_metrics.empty:
        return {
            "total_questions": 0,
            "overall_prt_elements": 0,
            "most_difficult_question": "-",
            "syntax_error_count": 0,
            "average_score": 0.0,
            "average_valid_submission_rate": 0.0,
            "average_correct_rate": 0.0,
            "student_count": 0,
        }

    most_difficult_question = question_metrics.sort_values(["scaled_score", "percent_correct", "attempts"], ascending=[True, True, False]).iloc[0]["question"] if not question_metrics.empty else "-"

    # Find total unique PRT index elements across all questions
    overall_prts = 0
    grouped = response_df.groupby("question")
    for _, rows in grouped:
        M = max([prt["index"] for prt_list in rows["prt_list"] for prt in prt_list] + [1])
        overall_prts += M

    return {
        "total_questions": int(question_metrics["question"].nunique()),
        "overall_prt_elements": overall_prts,
        "most_difficult_question": most_difficult_question,
        "syntax_error_count": int(question_metrics["syntax_error_count"].sum()),
        "average_score": round(float(question_metrics["avg_score"].mean()), 2),
        "average_valid_submission_rate": round(float(question_metrics["percent_valid"].mean()), 2),
        "average_correct_rate": round(float(question_metrics["percent_correct"].mean()), 2),
        "student_count": int(response_df["student_id"].nunique()),
    }


def compute_ranked_difficulty(question_metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank questions by average scaled score (lowest first)."""
    if question_metrics.empty:
        return pd.DataFrame(columns=["rank", "question", "average_score", "correct_percent", "attempts"])

    ranked = question_metrics.sort_values(["scaled_score", "percent_correct", "attempts"], ascending=[True, True, False]).reset_index(drop=True)
    ranked = ranked.rename(columns={"avg_score": "average_score", "percent_correct": "correct_percent"})
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked[["rank", "question", "average_score", "correct_percent", "attempts"]]
