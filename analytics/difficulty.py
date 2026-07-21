from __future__ import annotations

import re
import pandas as pd
from analytics.parser import get_attempt_pools


def compute_difficulty_metrics(response_df: pd.DataFrame) -> pd.DataFrame:
    """Compute difficulty, marks stats, and discrimination index D using Pool B (Best Attempt per Student)."""
    if response_df.empty or "question" not in response_df.columns:
        return pd.DataFrame(columns=[
            "question", "difficulty_index", "discrimination_index",
            "average_marks", "median_marks", "standard_deviation", "variance", "success_rate"
        ])

    _, pool_b_df = get_attempt_pools(response_df)

    def get_q_num(q_name: str) -> int:
        m = re.search(r"\d+", str(q_name))
        return int(m.group(0)) if m else 0

    questions = sorted(pool_b_df["question"].unique(), key=get_q_num)

    # Calculate overall quiz performance per student in Pool B for Top/Bottom 27% cohort ranking
    student_scores = pool_b_df.groupby("student_id")["overall_grade"].first()
    sorted_students = student_scores.sort_values(ascending=False).index.tolist()
    N_students = len(sorted_students)

    k = max(1, round(0.27 * N_students)) if N_students > 0 else 0
    top_group = set(sorted_students[:k]) if k > 0 else set()
    bottom_group = set(sorted_students[-k:]) if k > 0 else set()

    rows = []
    for q in questions:
        q_b = pool_b_df[pool_b_df["question"] == q]
        if q_b.empty:
            continue

        # Scores out of 10.0
        scores_10 = q_b["grade"] * 10.0
        avg_marks = float(scores_10.mean())
        median_marks = float(scores_10.median())
        std_marks = float(scores_10.std(ddof=1)) if len(scores_10) > 1 else 0.0
        var_marks = float(scores_10.var(ddof=1)) if len(scores_10) > 1 else 0.0

        facility = float((q_b["grade"] == 1.0).sum() / len(q_b)) if len(q_b) > 0 else 0.0
        success_rate = facility * 100.0

        # Discrimination Index D = Facility_top - Facility_bottom
        top_q = q_b[q_b["student_id"].isin(top_group)]
        bottom_q = q_b[q_b["student_id"].isin(bottom_group)]

        f_top = float((top_q["grade"] == 1.0).sum() / len(top_q)) if len(top_q) > 0 else 0.0
        f_bottom = float((bottom_q["grade"] == 1.0).sum() / len(bottom_q)) if len(bottom_q) > 0 else 0.0
        d_index = f_top - f_bottom

        rows.append({
            "question": q,
            "difficulty_index": round(success_rate, 2),
            "discrimination_index": round(d_index, 4),
            "average_marks": round(avg_marks, 2),
            "median_marks": round(median_marks, 2),
            "standard_deviation": round(std_marks, 2),
            "variance": round(var_marks, 2),
            "success_rate": round(success_rate, 2),
        })

    return pd.DataFrame(rows)
