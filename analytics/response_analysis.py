from __future__ import annotations

from collections import Counter
import re
import pandas as pd
from analytics.parser import get_attempt_pools


def compute_response_outcomes(response_df: pd.DataFrame) -> pd.DataFrame:
    """Compute response outcome percentages (Pool B for correct/incorrect, Pool A for valid/invalid)."""
    if response_df.empty or "question" not in response_df.columns:
        return pd.DataFrame(columns=["question", "correct_percent", "incorrect_percent", "valid_percent", "invalid_percent"])

    pool_a_df, pool_b_df = get_attempt_pools(response_df)

    def get_q_num(q_name: str) -> int:
        m = re.search(r"\d+", str(q_name))
        return int(m.group(0)) if m else 0

    questions = sorted(pool_a_df["question"].unique(), key=get_q_num)
    rows = []

    for q in questions:
        q_a = pool_a_df[pool_a_df["question"] == q]
        q_b = pool_b_df[pool_b_df["question"] == q]

        # Pool B: correct vs incorrect
        len_b = len(q_b)
        facility_b = (q_b["grade"] == 1.0).sum() / len_b if len_b > 0 else 0.0
        correct_pct = facility_b * 100.0
        incorrect_pct = (1.0 - facility_b) * 100.0

        # Pool A: valid vs invalid vs blank
        len_a = len(q_a)
        invalid_a = (q_a["response_status"] == "invalid").sum()
        blank_a = (q_a["response_status"] == "blank").sum()
        invalid_pct = (invalid_a / len_a * 100.0) if len_a > 0 else 0.0
        blank_pct = (blank_a / len_a * 100.0) if len_a > 0 else 0.0
        valid_pct = max(0.0, 100.0 - invalid_pct - blank_pct)

        rows.append({
            "question": q,
            "correct_percent": round(correct_pct, 2),
            "incorrect_percent": round(incorrect_pct, 2),
            "valid_percent": round(valid_pct, 2),
            "invalid_percent": round(invalid_pct, 2),
        })

    return pd.DataFrame(rows)


def compute_repeated_wrong_answers(response_df: pd.DataFrame) -> pd.DataFrame:
    """Tally most frequent wrong literal inputs strictly from Pool B (Best Attempt per Student)."""
    if response_df.empty or "question" not in response_df.columns:
        return pd.DataFrame(columns=["question", "most_common_incorrect_answer", "frequency"])

    _, pool_b_df = get_attempt_pools(response_df)

    def get_q_num(q_name: str) -> int:
        m = re.search(r"\d+", str(q_name))
        return int(m.group(0)) if m else 0

    questions = sorted(pool_b_df["question"].unique(), key=get_q_num)
    rows = []

    for q in questions:
        wrong_b = pool_b_df[(pool_b_df["question"] == q) & (pool_b_df["grade"] < 1.0)]

        expr_counts: Counter = Counter()
        for _, r in wrong_b.iterrows():
            ans_list = r.get("ans_list") or []
            for ans in ans_list:
                tag = ans.get("tag")
                expr = str(ans.get("expression", "")).strip()
                if expr and tag in ("invalid", "valid", "score"):
                    expr_counts[expr] += 1

        if expr_counts:
            top_wrong = expr_counts.most_common(5)
            formatted_list = [f"{expr} ({cnt})" for expr, cnt in top_wrong]
            most_common_str = ", ".join(formatted_list)
            top_freq = top_wrong[0][1]
        else:
            top_wrong = []
            most_common_str = "None"
            top_freq = 0

        rows.append({
            "question": q,
            "most_common_incorrect_answer": most_common_str,
            "frequency": top_freq,
            "top_wrong_expressions": top_wrong,
        })

    return pd.DataFrame(rows)
