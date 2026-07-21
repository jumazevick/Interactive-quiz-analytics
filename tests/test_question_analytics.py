import os
import pandas as pd
from analytics.parser import parse_response_cell, build_response_rows
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


def test_build_response_rows_filters_in_progress():
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
            "State": "In progress",
            "Grade/10.00": "-",
            "Response 1": "-",
        }
    ])
    res_df = build_response_rows(df, "Quiz 1")
    assert len(res_df) == 1
    assert res_df.iloc[0]["student_id"] == "jane@example.com"
    assert res_df.iloc[0]["grade"] == 1.0
    assert res_df.iloc[0]["response_status"] == "correct"


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

