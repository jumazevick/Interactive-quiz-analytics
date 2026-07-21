from __future__ import annotations

import pandas as pd


def compute_syntax_analysis(response_df: pd.DataFrame) -> pd.DataFrame:
    """Compute syntax-error metrics and common mistakes per question."""
    if response_df.empty:
        return pd.DataFrame(columns=["question", "syntax_error_count", "percentage", "common_mistakes"])

    summary = []
    for question, group in response_df.groupby("question"):
        syntax_rows = group[group["response_status"] == "invalid"]
        count = int(len(syntax_rows))
        percentage = round(float(count / len(group) * 100), 2) if len(group) else 0.0

        # Collect invalid expressions
        invalid_exprs = []
        for _, row in syntax_rows.iterrows():
            ans_list = row.get("ans_list", [])
            for ans in ans_list:
                if ans["tag"] == "invalid" and ans["expression"]:
                    invalid_exprs.append(ans["expression"])

        # Top 3 mistakes
        from collections import Counter
        counts = Counter(invalid_exprs)
        most_common = [expr for expr, _ in counts.most_common(3)]
        common_mistakes = ", ".join(most_common) if most_common else "-"

        summary.append({
            "question": question,
            "syntax_error_count": count,
            "percentage": percentage,
            "common_mistakes": common_mistakes
        })

    return pd.DataFrame(summary).sort_values("question").reset_index(drop=True)
