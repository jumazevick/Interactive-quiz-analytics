import io
import os
import re
import pandas as pd
from analytics.parser import (
    parse_response_cell,
    build_response_rows,
    get_attempt_pools,
    detect_export_type,
    build_grade_breakdown_rows,
    merge_grade_breakdown_rows,
)
from analytics.pdf_export import generate_pdf_report
from pages.Question_Analysis_Section import build_question_analytics


def test_parse_response_cell():
    cell = "Seed: 2041245669; ans1: 2 [score]; ans2: 0 [valid]; prt1: # = 1 | prt1-1-T; prt2: # = 0 | prt2-1-F"
    ans_list, prt_list = parse_response_cell(cell)
    assert len(ans_list) == 2
    assert ans_list[0]["expression"] == "2"
    assert ans_list[0]["tag"] == "score"
    assert ans_list[1]["tag"] == "valid"
    assert len(prt_list) == 2
    assert prt_list[0]["fraction"] == 1.0
    assert prt_list[0]["answer_note"] == "prt1-1-T"
    assert prt_list[1]["fraction"] == 0.0


def test_get_attempt_pools_selects_best_attempt():
    df = pd.DataFrame([
        # Student 1: Attempt 1 (Grade 5.0)
        {"student_id": "s1", "question": "Q1", "grade": 0.5, "overall_grade": 5.0, "completed_dt": pd.Timestamp("2026-07-21 10:00"), "attempt_idx": 1},
        {"student_id": "s1", "question": "Q2", "grade": 0.5, "overall_grade": 5.0, "completed_dt": pd.Timestamp("2026-07-21 10:00"), "attempt_idx": 1},
        # Student 1: Attempt 2 (Grade 10.0 - Best)
        {"student_id": "s1", "question": "Q1", "grade": 1.0, "overall_grade": 10.0, "completed_dt": pd.Timestamp("2026-07-21 11:00"), "attempt_idx": 2},
        {"student_id": "s1", "question": "Q2", "grade": 1.0, "overall_grade": 10.0, "completed_dt": pd.Timestamp("2026-07-21 11:00"), "attempt_idx": 2},
        # Student 2: Single Attempt
        {"student_id": "s2", "question": "Q1", "grade": 0.0, "overall_grade": 0.0, "completed_dt": pd.Timestamp("2026-07-21 10:30"), "attempt_idx": 3},
        {"student_id": "s2", "question": "Q2", "grade": 0.0, "overall_grade": 0.0, "completed_dt": pd.Timestamp("2026-07-21 10:30"), "attempt_idx": 3},
    ])

    pool_a, pool_b = get_attempt_pools(df)
    assert len(pool_a) == 6  # 3 attempts * 2 questions = 6 rows
    assert len(pool_b) == 4  # 2 best attempts * 2 questions = 4 rows

    s1_b_grades = pool_b[pool_b["student_id"] == "s1"]["grade"].tolist()
    assert s1_b_grades == [1.0, 1.0]


def test_pdf_report_generation():
    df = pd.DataFrame([{"Question": "Q1", "Attempts": 34, "Facility": 0.74}])
    pdf_bytes = generate_pdf_report(
        title="Test Report",
        subtitle="Subtitle",
        sections=[{"title": "Section 1", "caption": "Caption", "df": df, "notes": ["Note 1"]}],
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_question_analytics_structure():
    df = pd.DataFrame([
        {
            "Surname": "Doe",
            "First name": "Jane",
            "Email address": "jane@example.com",
            "State": "Finished",
            "Grade/10.00": "10.00",
            "Response 1": "Seed: 1; ans1: 5 [score]; prt1: # = 1 | prt1-1-T",
        },
        {
            "Surname": "Smith",
            "First name": "John",
            "Email address": "john@example.com",
            "State": "Finished",
            "Grade/10.00": "0.00",
            "Response 1": "Seed: 2; ans1: 0 [score]; prt1: # = 0 | prt1-1-F",
        }
    ])
    analytics = build_question_analytics(df, quiz_name="Quiz 1")
    assert analytics["question_metrics"].shape[0] == 1
    assert analytics["question_summary"]["total_questions"] == 1
    assert analytics["question_summary"]["student_count"] == 2


def test_detect_grades_breakdown_export_and_merge_scores():
    df = pd.DataFrame([
        {
            "Last name": "Doe",
            "First name": "Jane",
            "Email address": "jane@example.com",
            "State": "Finished",
            "Started on": "2026-07-22 09:00:00",
            "Completed": "2026-07-22 09:05:00",
            "Grade/10.00": "5.00",
            "Q. 1 /2.50": "1.25",
            "Q. 2 /2.50": "0.00",
        }
    ])

    assert detect_export_type(df) == "grades_breakdown"

    grade_rows = build_grade_breakdown_rows(df, quiz_name="Quiz 1")
    assert grade_rows.loc[0, "question"] == "Q1"
    assert grade_rows.loc[0, "grade"] == 0.5
    assert grade_rows.loc[1, "grade"] == 0.0

    response_rows = build_response_rows(pd.DataFrame([
        {
            "Last name": "Doe",
            "First name": "Jane",
            "Email address": "jane@example.com",
            "State": "Finished",
            "Started on": "2026-07-22 09:00:00",
            "Completed": "2026-07-22 09:05:00",
            "Grade/10.00": "5.00",
            "Response 1": "ans1: 1 [score]; prt1: # = 1 | prt1-1-T",
            "Response 2": "ans2: 0 [score]; prt2: # = 0 | prt2-1-F",
        }
    ]), quiz_name="Quiz 1")

    merged = merge_grade_breakdown_rows(response_rows, grade_rows)
    assert merged.loc[0, "grade"] == 0.5
    assert merged.loc[0, "response_text"].startswith("ans1")
    assert merged.loc[1, "grade"] == 0.0


def test_student_matrix_keeps_all_students_and_questions_despite_blank_cells():
    # Round 4 regression test: a blank response for one question must never drop that
    # student's row or that question's column out of the Section 4 pivot matrix.
    rows = []
    for i in range(6):
        # Every third student leaves Q2 blank (empty response cell -> grade 0.0, not NaN).
        r2 = "" if i % 3 == 0 else "ans1: 1 [score]; prt1: # = 1 | prt1-1-T"
        rows.append({
            "Surname": f"S{i}", "First name": "Test", "Email address": f"student{i}@example.com",
            "State": "Finished", "Grade/10.00": "5.00",
            "Response 1": "ans1: 1 [score]; prt1: # = 1 | prt1-1-T",
            "Response 2": r2,
        })
    df = pd.DataFrame(rows)
    analytics = build_question_analytics(df, quiz_name="Quiz 1")
    pool_b_df = analytics["pool_b_df"]

    assert not pool_b_df["grade"].isna().any()

    question_order = sorted(pool_b_df["question"].unique(), key=lambda q: int(re.search(r"\d+", str(q)).group(0)))
    matrix = pool_b_df.pivot_table(
        index="student_id", columns="question", values="grade",
        aggfunc="first", fill_value=0.0, dropna=False,
    ).reindex(columns=question_order, fill_value=0.0)

    assert matrix.shape == (pool_b_df["student_id"].nunique(), len(question_order))
    assert matrix.shape == (6, 2)


def _tiny_png_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color="white").save(buf, format="PNG")
    return buf.getvalue()


def test_pdf_report_embeds_chart_images():
    png_bytes = _tiny_png_bytes()

    class FakePlotlyFigure:
        def to_image(self, format="png", scale=2):
            assert format == "png"
            return png_bytes

    class FakeMatplotlibFigure:
        def savefig(self, buf, format="png", dpi=150, bbox_inches="tight"):
            buf.write(png_bytes)

    df = pd.DataFrame([{"Question": "Q1", "Attempts": 34, "Facility": 0.74}])
    pdf_bytes = generate_pdf_report(
        title="Test Report",
        subtitle="Subtitle",
        sections=[{
            "title": "Section 1",
            "caption": "Caption",
            "df": df,
            "charts": [
                {"title": "Plotly Chart", "figure": FakePlotlyFigure()},
                {"title": "Matplotlib Chart", "figure": FakeMatplotlibFigure()},
            ],
        }],
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


