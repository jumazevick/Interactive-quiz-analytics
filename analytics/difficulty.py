from __future__ import annotations

import pandas as pd


def compute_difficulty_metrics(response_df: pd.DataFrame) -> pd.DataFrame:
    """Compute difficulty and discrimination metrics per question."""
    if response_df.empty:
        return pd.DataFrame(columns=["question", "difficulty_index", "discrimination_index", "average_marks", "median_marks", "standard_deviation", "variance", "success_rate"])

    summary = []
    for question, group in response_df.groupby("question"):
        scores = group["grade"].astype(float).to_numpy()
        max_grade = group["max_grade"].replace(0, 1).astype(float).to_numpy()
        normalized_scores = scores / max_grade
        normalized_scores = normalized_scores * 10.0

        average_marks = round(float(normalized_scores.mean()), 2) if len(normalized_scores) else 0.0
        median_marks = round(float(pd.Series(normalized_scores).median()), 2) if len(normalized_scores) else 0.0
        std = round(float(pd.Series(normalized_scores).std()), 2) if len(normalized_scores) > 1 else 0.0
        variance = round(float(pd.Series(normalized_scores).var()), 2) if len(normalized_scores) > 1 else 0.0
        success_rate = round(float((group["response_status"] == "correct").mean()) * 100, 2) if len(group) else 0.0

        difficulty_index = round(float(100 - success_rate), 2)
        discrimination_index = round(float((success_rate - (100 - success_rate)) / 100), 2)

        summary.append({
            "question": question,
            "difficulty_index": difficulty_index,
            "discrimination_index": discrimination_index,
            "average_marks": average_marks,
            "median_marks": median_marks,
            "standard_deviation": std,
            "variance": variance,
            "success_rate": success_rate,
        })

    return pd.DataFrame(summary).sort_values("question").reset_index(drop=True)
