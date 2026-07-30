from __future__ import annotations

import pandas as pd

from analytics.difficulty import compute_difficulty_metrics
from analytics.parser import (
    build_grade_breakdown_rows,
    build_response_rows,
    detect_export_type,
    get_attempt_pools,
)
from analytics.prt_analysis import build_prt_frame, compute_prt_pass_rates
from analytics.question_metrics import compute_question_metrics, compute_question_summary, compute_ranked_difficulty
from analytics.response_analysis import compute_repeated_wrong_answers, compute_response_outcomes
from analytics.summary import build_export_summary
from analytics.syntax_analysis import compute_syntax_analysis


def build_question_analytics(response_df: pd.DataFrame, quiz_name: str) -> dict[str, object]:
    """Every question-level analytic for one quiz, in one dict.

    Lives here rather than on the page that renders it so the page files (whose names
    start with a digit, for nav ordering, and so aren't importable as modules) aren't the
    only place it exists — the Question Analysis page, the shared PDF report builder, and
    the test suite all call it.
    """
    if "question" not in response_df.columns:
        export_type = detect_export_type(response_df)
        if export_type == "grades_breakdown":
            response_df = build_grade_breakdown_rows(response_df, quiz_name=quiz_name)
        else:
            response_df = build_response_rows(response_df, quiz_name=quiz_name)

    pool_a_df, pool_b_df = get_attempt_pools(response_df)

    question_metrics = compute_question_metrics(response_df)
    prt_frame = build_prt_frame(pool_a_df)
    question_summary = compute_question_summary(response_df, prt_frame)
    response_outcomes = compute_response_outcomes(response_df)
    difficulty_metrics = compute_difficulty_metrics(response_df)
    syntax_analysis = compute_syntax_analysis(pool_a_df)
    prt_pass_rates = compute_prt_pass_rates(prt_frame)
    repeated_wrong_answers = compute_repeated_wrong_answers(response_df)
    ranked_difficulty = compute_ranked_difficulty(question_metrics)
    export_summary = build_export_summary(question_metrics, response_outcomes, difficulty_metrics, syntax_analysis, prt_pass_rates, repeated_wrong_answers)

    return {
        "question_metrics": question_metrics,
        "question_summary": question_summary,
        "response_outcomes": response_outcomes,
        "difficulty_metrics": difficulty_metrics,
        "syntax_analysis": syntax_analysis,
        "prt_frame": prt_frame,
        "prt_pass_rates": prt_pass_rates,
        "repeated_wrong_answers": repeated_wrong_answers,
        "ranked_difficulty": ranked_difficulty,
        "export_summary": export_summary,
        "quiz_name": quiz_name,
        "pool_a_df": pool_a_df,
        "pool_b_df": pool_b_df,
    }
