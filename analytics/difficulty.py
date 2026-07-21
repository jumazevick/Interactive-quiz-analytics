from __future__ import annotations

import pandas as pd


def compute_difficulty_metrics(response_df: pd.DataFrame) -> pd.DataFrame:
    """Compute difficulty and discrimination metrics per question."""
    if response_df.empty:
        return pd.DataFrame(columns=["question", "difficulty_index", "discrimination_index", "average_marks", "median_marks", "standard_deviation", "variance", "success_rate"])

    # 1. Compute each student's highest overall quiz score (for ranking)
    # response_df contains student_id and overall_grade.
    student_best_overall = response_df.groupby("student_id")["overall_grade"].max().to_dict()
    sorted_students = sorted(student_best_overall.keys(), key=lambda s: student_best_overall[s], reverse=True)
    num_students = len(sorted_students)

    # Top & Bottom 27% cohorts
    cohort_size = max(1, int(round(num_students * 0.27)))
    top_students = set(sorted_students[:cohort_size])
    bottom_students = set(sorted_students[-cohort_size:])

    summary = []
    for question, group in response_df.groupby("question"):
        # For each student, get their best question score for this question (max grade)
        student_best_q = group.groupby("student_id")["grade"].max().to_dict()

        # Calculate F_top and F_bottom
        f_top_correct = sum(1 for s in top_students if student_best_q.get(s, 0.0) == 1.0)
        F_top = f_top_correct / len(top_students) if top_students else 0.0

        f_bottom_correct = sum(1 for s in bottom_students if student_best_q.get(s, 0.0) == 1.0)
        F_bottom = f_bottom_correct / len(bottom_students) if bottom_students else 0.0

        D = round(float(F_top - F_bottom), 2)

        # Marks (grade * 10.0 scale)
        scores_x10 = group["grade"].astype(float).to_numpy() * 10.0
        average_marks = round(float(scores_x10.mean()), 2) if len(scores_x10) else 0.0
        median_marks = round(float(pd.Series(scores_x10).median()), 2) if len(scores_x10) else 0.0
        std = round(float(pd.Series(scores_x10).std()), 2) if len(scores_x10) > 1 else 0.0
        variance = round(float(pd.Series(scores_x10).var()), 2) if len(scores_x10) > 1 else 0.0

        # facility of this question computed over all attempts
        attempts = len(group)
        facility = (group["grade"] == 1.0).sum() / attempts if attempts else 0.0
        success_rate = round(float(facility * 100), 2)
        difficulty_index = success_rate # as facility * 100

        summary.append({
            "question": question,
            "difficulty_index": difficulty_index,
            "discrimination_index": D,
            "average_marks": average_marks,
            "median_marks": median_marks,
            "standard_deviation": std,
            "variance": variance,
            "success_rate": success_rate,
        })

    return pd.DataFrame(summary).sort_values("question").reset_index(drop=True)
