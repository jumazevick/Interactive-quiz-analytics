import pandas as pd

from pages.Question_Analysis_Section import build_question_analytics


def test_build_question_analytics_returns_expected_structure():
    df = pd.DataFrame(
        [
            {
                "Surname": "Doe",
                "First name": "Jane",
                "Email address": "jane@example.com",
                "State": "Finished",
                "Response 1": "prt1: # = 1;",
                "Q1": 1.0,
                "Q1 state": "finished",
            },
            {
                "Surname": "Smith",
                "First name": "John",
                "Email address": "john@example.com",
                "State": "Finished",
                "Response 1": "prt1: # = 0;",
                "Q1": 0.0,
                "Q1 state": "finished",
            },
        ]
    )

    analytics = build_question_analytics(df, quiz_name="Quiz 1")

    assert analytics["question_metrics"].shape[0] == 1
    assert analytics["question_summary"]["total_questions"] == 1
    assert analytics["question_summary"]["syntax_error_count"] == 0
    assert analytics["question_summary"]["student_count"] == 2
