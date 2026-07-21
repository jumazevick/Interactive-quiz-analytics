from __future__ import annotations

import pandas as pd


def compute_response_outcomes(response_df: pd.DataFrame) -> pd.DataFrame:
    """Compute response outcome percentages per question."""
    if response_df.empty:
        return pd.DataFrame(columns=["question", "correct_percent", "incorrect_percent", "valid_percent", "invalid_percent"])

    summary = []
    for question, group in response_df.groupby("question"):
        attempts = len(group)
        correct_percent = round(float((group["response_status"] == "correct").mean()) * 100, 2) if attempts else 0.0
        incorrect_percent = round(float((group["response_status"] == "incorrect").mean()) * 100, 2) if attempts else 0.0
        valid_percent = round(float(((group["response_status"] != "invalid") & (group["response_status"] != "syntax_error") & (group["response_status"] != "blank")).mean()) * 100, 2) if attempts else 0.0
        invalid_percent = round(float(((group["response_status"] == "invalid") | (group["response_status"] == "syntax_error")).mean()) * 100, 2) if attempts else 0.0
        summary.append({
            "question": question,
            "correct_percent": correct_percent,
            "incorrect_percent": incorrect_percent,
            "valid_percent": valid_percent,
            "invalid_percent": invalid_percent,
        })

    return pd.DataFrame(summary).sort_values("question").reset_index(drop=True)


def compute_repeated_wrong_answers(response_df: pd.DataFrame) -> pd.DataFrame:
    """Identify repeated incorrect responses for each question."""
    if response_df.empty:
        return pd.DataFrame(columns=["question", "most_common_incorrect_answer", "frequency", "percentage"])

    summary = []
    for question, group in response_df.groupby("question"):
        wrong_rows = group[group["response_status"].isin(["incorrect", "invalid", "syntax_error"]) & (group["response_text"].astype(str).str.strip() != "")]
        if wrong_rows.empty:
            summary.append({"question": question, "most_common_incorrect_answer": "-", "frequency": 0, "percentage": 0.0})
            continue

        counts = wrong_rows["response_text"].astype(str).value_counts()
        answer, frequency = counts.idxmax(), int(counts.max())
        percentage = round(float(frequency / len(wrong_rows) * 100), 2)
        summary.append({
            "question": question,
            "most_common_incorrect_answer": answer,
            "frequency": frequency,
            "percentage": percentage,
        })

    return pd.DataFrame(summary).sort_values("question").reset_index(drop=True)
