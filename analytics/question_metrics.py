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
                "avg_score",
                "percent_correct",
                "percent_incorrect",
                "percent_valid",
                "percent_invalid",
                "syntax_error_count",
                "syntax_error_percent",
            ]
        )

    grouped = response_df.groupby("question")
    metrics = []
    for question, rows in grouped:
        attempts = len(rows)
        avg_score = float(rows["grade"].mean()) if attempts else 0.0
        raw_score = rows["grade"].to_numpy(dtype=float)
        max_grade = rows["max_grade"].replace(0, np.nan).to_numpy(dtype=float)
        scaled_scores = np.divide(raw_score, max_grade, out=np.zeros_like(raw_score, dtype=float), where=max_grade > 0)
        scaled_scores *= 10.0

        correct_count = int((rows["response_status"] == "correct").sum())
        incorrect_count = int((rows["response_status"] == "incorrect").sum())
        syntax_error_count = int((rows["response_status"] == "syntax_error").sum())
        invalid_count = int(((rows["response_status"] == "invalid") | (rows["response_status"] == "syntax_error")).sum())
        blank_count = int((rows["response_status"] == "blank").sum())

        percent_correct = round(correct_count / attempts * 100, 2) if attempts else 0.0
        percent_incorrect = round(incorrect_count / attempts * 100, 2) if attempts else 0.0
        percent_valid = round((attempts - invalid_count - blank_count) / attempts * 100, 2) if attempts else 0.0
        percent_invalid = round(invalid_count / attempts * 100, 2) if attempts else 0.0
        syntax_error_percent = round(syntax_error_count / attempts * 100, 2) if attempts else 0.0

        metrics.append(
            {
                "question": question,
                "attempts": attempts,
                "avg_score": round(float(avg_score), 2),
                "percent_correct": percent_correct,
                "percent_incorrect": percent_incorrect,
                "percent_valid": percent_valid,
                "percent_invalid": percent_invalid,
                "syntax_error_count": syntax_error_count,
                "syntax_error_percent": syntax_error_percent,
                "scaled_score": round(float(scaled_scores.mean()), 2) if attempts else 0.0,
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

    prt_element_count = int(prt_frame.shape[0]) if prt_frame is not None and not prt_frame.empty else int(question_metrics.shape[0])

    return {
        "total_questions": int(question_metrics["question"].nunique()),
        "overall_prt_elements": prt_element_count,
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
