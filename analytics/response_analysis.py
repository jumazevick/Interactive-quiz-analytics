from __future__ import annotations

import pandas as pd


def compute_response_outcomes(response_df: pd.DataFrame) -> pd.DataFrame:
    """Compute response outcome percentages per question."""
    if response_df.empty:
        return pd.DataFrame(columns=["question", "correct_percent", "incorrect_percent", "valid_percent", "invalid_percent"])

    summary = []
    for question, group in response_df.groupby("question"):
        attempts = len(group)
        correct_count = int((group["response_status"] == "correct").sum())
        invalid_count = int((group["response_status"] == "invalid").sum())
        blank_count = int((group["response_status"] == "blank").sum())

        facility = correct_count / attempts if attempts else 0.0
        invalid_rate = invalid_count / attempts if attempts else 0.0
        blank_rate = blank_count / attempts if attempts else 0.0

        correct_percent = round(facility * 100, 2)
        incorrect_percent = round((1 - facility) * 100, 2)
        valid_percent = round((1 - invalid_rate - blank_rate) * 100, 2)
        invalid_percent = round(invalid_rate * 100, 2)

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
        wrong_rows = group[group["grade"] < 1.0]
        if wrong_rows.empty:
            summary.append({
                "question": question,
                "most_common_incorrect_answer": "-",
                "frequency": 0,
                "percentage": 0.0
            })
            continue

        # Collect wrong expressions
        wrong_exprs = []
        for _, row in wrong_rows.iterrows():
            ans_list = row.get("ans_list", [])
            prt_list = row.get("prt_list", [])
            prt_map = {prt["index"]: prt for prt in prt_list}
            for ans in ans_list:
                k = ans["index"]
                tag = ans["tag"]
                expr = ans["expression"]
                is_wrong = False
                if tag == "invalid":
                    is_wrong = True
                else:
                    prt = prt_map.get(k)
                    if prt is None or prt["fraction"] is None or prt["fraction"] < 1.0:
                        is_wrong = True

                if is_wrong and expr:
                    wrong_exprs.append(expr)

        if not wrong_exprs:
            summary.append({
                "question": question,
                "most_common_incorrect_answer": "-",
                "frequency": 0,
                "percentage": 0.0
            })
            continue

        # Count frequencies
        from collections import Counter
        counts = Counter(wrong_exprs)
        most_common = counts.most_common(5)  # top 5

        # Format top 5 wrong inputs
        formatted_ans = ", ".join([f"'{expr}' (x{count})" for expr, count in most_common])

        # Frequency and percentage of the single most common
        top_expr, top_count = most_common[0]
        percentage = round(float(top_count / len(wrong_rows) * 100), 2)

        summary.append({
            "question": question,
            "most_common_incorrect_answer": formatted_ans,
            "frequency": top_count,
            "percentage": percentage,
        })

    return pd.DataFrame(summary).sort_values("question").reset_index(drop=True)
