from __future__ import annotations

import pandas as pd


def compute_syntax_analysis(response_df: pd.DataFrame) -> pd.DataFrame:
    """Compute syntax-error metrics and common mistakes per question."""
    if response_df.empty:
        return pd.DataFrame(columns=["question", "syntax_error_count", "percentage", "common_mistakes"])

    summary = []
    for question, group in response_df.groupby("question"):
        syntax_rows = group[group["response_status"] == "syntax_error"]
        count = int(len(syntax_rows))
        percentage = round(float(count / len(group) * 100), 2) if len(group) else 0.0
        mistakes = []
        for text in syntax_rows["response_text"].astype(str).tolist():
            if text.strip() and text.strip() != "!":
                mistakes.append(text)
        common_mistakes = ", ".join(mistakes[:3]) if mistakes else "-"
        summary.append({"question": question, "syntax_error_count": count, "percentage": percentage, "common_mistakes": common_mistakes})

    return pd.DataFrame(summary).sort_values("question").reset_index(drop=True)
