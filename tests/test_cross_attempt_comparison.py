import pandas as pd

from analytics.parser import parse_response_cell
from analytics.solution_distance import (
    CROSS_ATTEMPT_METRICS,
    build_cross_attempt_figure,
    build_single_student_attempt_figure,
    classify_cross_attempt_trends,
    compute_cross_attempt_comparison,
)

RIGHT_ANSWER = "ans1: -1/(2*x*sqrt(x)) [valid]"
CORRECT = "Seed: 1; ans1: -1/(2*x*sqrt(x)) [score]; prt1: # = 1 | prt1-1-T"
NODE2_TRUE = "Seed: 1; ans1: 1/(2*sqrt(x)) [score]; prt1: # = 0.5 | prt1-1-F | prt1-2-T"
ALL_FALSE = "Seed: 1; ans1: 7 [score]; prt1: # = 0 | prt1-1-F | prt1-2-F | prt1-3-F"


def _row(student_id, attempt_idx, cell, question="Q1", grade=None, right_answer=RIGHT_ANSWER):
    ans_list, prt_list = parse_response_cell(cell)
    return {
        "question": question,
        "student_id": student_id,
        "student_name": student_id.upper(),
        "response_status": "incorrect",
        "prt_list": prt_list,
        "ans_list": ans_list,
        "right_answer_text": right_answer,
        "grade": grade if grade is not None else 0.0,
        "overall_grade": 0.0,
        "completed_dt": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=attempt_idx),
        "attempt_idx": attempt_idx,
    }


def test_students_with_a_single_attempt_are_dropped():
    """A single attempt has no 'change' to show, per the module's own stated scope."""
    df = pd.DataFrame([
        _row("s1", 1, ALL_FALSE, grade=0.0), _row("s1", 2, CORRECT, grade=1.0),  # 2 attempts: kept
        _row("s2", 1, CORRECT, grade=1.0),                                       # 1 attempt: dropped
    ])
    comparison = compute_cross_attempt_comparison(df, "Q1", "Grade")
    assert set(comparison["student_id"]) == {"s1"}


def test_grade_metric_uses_the_0_to_10_display_scale():
    df = pd.DataFrame([_row("s1", 1, ALL_FALSE, grade=0.0), _row("s1", 2, CORRECT, grade=1.0)])
    comparison = compute_cross_attempt_comparison(df, "Q1", "Grade")
    values = comparison.sort_values("attempt_number")["value"].tolist()
    assert values == [0.0, 10.0]


def test_attempt_number_is_sequential_per_student_not_global():
    df = pd.DataFrame([
        _row("s1", 1, ALL_FALSE, grade=0.0), _row("s1", 2, NODE2_TRUE, grade=0.5), _row("s1", 3, CORRECT, grade=1.0),
    ])
    comparison = compute_cross_attempt_comparison(df, "Q1", "Grade").sort_values("attempt_number")
    assert comparison["attempt_number"].tolist() == [1, 2, 3]


def test_empty_result_has_the_documented_columns():
    empty = compute_cross_attempt_comparison(pd.DataFrame(), "Q1", "Grade")
    assert list(empty.columns) == ["student_id", "student_name", "attempt_number", "value", "completed_dt"]
    assert empty.empty


def test_comparison_rows_carry_a_completed_dt_per_attempt():
    """The per-attempt drill-down view needs a date to show alongside each attempt's value."""
    df = pd.DataFrame([_row("s1", 1, ALL_FALSE, grade=0.0), _row("s1", 2, CORRECT, grade=1.0)])
    comparison = compute_cross_attempt_comparison(df, "Q1", "Grade").sort_values("attempt_number")
    assert comparison["completed_dt"].tolist() == [
        pd.Timestamp("2026-01-01") + pd.Timedelta(hours=1),
        pd.Timestamp("2026-01-01") + pd.Timedelta(hours=2),
    ]


def test_unknown_metric_raises():
    df = pd.DataFrame([_row("s1", 1, ALL_FALSE), _row("s1", 2, CORRECT)])
    try:
        compute_cross_attempt_comparison(df, "Q1", "Not A Real Metric")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_trend_classification_uses_first_vs_last_not_the_full_sequence():
    """A student who dipped in the middle but ended where they started is Flat, not
    Improved or Regressed -- only the first and last qualifying attempt matter."""
    df = pd.DataFrame([
        _row("s1", 1, NODE2_TRUE, grade=0.5),
        _row("s1", 2, ALL_FALSE, grade=0.0),
        _row("s1", 3, NODE2_TRUE, grade=0.5),
    ])
    comparison = compute_cross_attempt_comparison(df, "Q1", "Grade")
    trends = classify_cross_attempt_trends(comparison, higher_is_better=True)
    assert trends.iloc[0]["trend"] == "Flat"


def test_trend_direction_flips_correctly_for_a_lower_is_better_metric():
    """For a distance metric, a numeric decrease is an improvement -- the opposite of
    Grade's own higher-is-better sense -- and `change` must still read positive for it."""
    df = pd.DataFrame([
        _row("s1", 1, ALL_FALSE), _row("s1", 2, CORRECT),  # PRT distance: other(6) -> 0
    ])
    comparison = compute_cross_attempt_comparison(df, "Q1", "PRT Distance")
    trends = classify_cross_attempt_trends(comparison, higher_is_better=False)
    row = trends.iloc[0]
    assert row["last_value"] < row["first_value"]
    assert row["change"] > 0, "a numeric decrease on a lower-is-better metric must read as positive improvement"
    assert row["trend"] == "Improved"


def test_ranking_is_sorted_most_improved_first():
    df = pd.DataFrame([
        _row("regressed", 1, CORRECT, grade=1.0), _row("regressed", 2, ALL_FALSE, grade=0.0),
        _row("big_improver", 1, ALL_FALSE, grade=0.0), _row("big_improver", 2, CORRECT, grade=1.0),
        _row("flat", 1, NODE2_TRUE, grade=0.5), _row("flat", 2, NODE2_TRUE, grade=0.5),
    ])
    comparison = compute_cross_attempt_comparison(df, "Q1", "Grade")
    trends = classify_cross_attempt_trends(comparison, higher_is_better=True)
    assert trends["student_id"].tolist() == ["big_improver", "flat", "regressed"]


def test_tiny_float_drift_counts_as_flat_not_a_false_trend():
    df = pd.DataFrame([
        _row("s1", 1, CORRECT, grade=1.0), _row("s1", 2, CORRECT, grade=1.0 + 1e-12),
    ])
    comparison = compute_cross_attempt_comparison(df, "Q1", "Grade")
    trends = classify_cross_attempt_trends(comparison, higher_is_better=True)
    assert trends.iloc[0]["trend"] == "Flat"


def test_figure_legend_stays_at_three_entries_regardless_of_student_count():
    """One trace per student is fine for hover/lines, but the legend itself must stay at
    exactly the trend categories present -- never one entry per student."""
    rows = []
    for i in range(12):
        rows.append(_row(f"s{i}", 1, ALL_FALSE, grade=0.0))
        rows.append(_row(f"s{i}", 2, CORRECT, grade=1.0))
    df = pd.DataFrame(rows)
    comparison = compute_cross_attempt_comparison(df, "Q1", "Grade")
    trends = classify_cross_attempt_trends(comparison, higher_is_better=True)
    fig = build_cross_attempt_figure(comparison, trends, "Grade", colorblind_mode=False)
    assert len(fig.data) == 12, "one trace per student for correct hover/line rendering"
    shown = [t for t in fig.data if t.showlegend]
    assert len(shown) <= 3
    assert {t.name for t in shown}.issubset({"Improved", "Flat", "Regressed"})


def test_colorblind_mode_changes_the_trend_palette():
    df = pd.DataFrame([_row("s1", 1, ALL_FALSE, grade=0.0), _row("s1", 2, CORRECT, grade=1.0)])
    comparison = compute_cross_attempt_comparison(df, "Q1", "Grade")
    trends = classify_cross_attempt_trends(comparison, higher_is_better=True)
    normal = build_cross_attempt_figure(comparison, trends, "Grade", colorblind_mode=False)
    safe = build_cross_attempt_figure(comparison, trends, "Grade", colorblind_mode=True)
    assert normal.data[0].line.color != safe.data[0].line.color


def test_all_three_metrics_are_computable_end_to_end():
    df = pd.DataFrame([_row("s1", 1, ALL_FALSE, grade=0.0), _row("s1", 2, CORRECT, grade=1.0)])
    for metric in CROSS_ATTEMPT_METRICS:
        comparison = compute_cross_attempt_comparison(df, "Q1", metric, part_index=1)
        assert not comparison.empty, f"{metric} produced no comparable attempts"


def test_single_student_figure_plots_every_attempt_for_just_that_student():
    df = pd.DataFrame([
        _row("s1", 1, ALL_FALSE, grade=0.0),
        _row("s1", 2, NODE2_TRUE, grade=0.5),
        _row("s1", 3, CORRECT, grade=1.0),
    ])
    comparison = compute_cross_attempt_comparison(df, "Q1", "Grade")
    fig = build_single_student_attempt_figure(comparison, "S1", "Grade", "Improved", colorblind_mode=False)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [1, 2, 3]
    assert list(fig.data[0].y) == [0.0, 5.0, 10.0]


def test_single_student_figure_color_follows_colorblind_mode():
    df = pd.DataFrame([_row("s1", 1, ALL_FALSE, grade=0.0), _row("s1", 2, CORRECT, grade=1.0)])
    comparison = compute_cross_attempt_comparison(df, "Q1", "Grade")
    normal = build_single_student_attempt_figure(comparison, "S1", "Grade", "Improved", colorblind_mode=False)
    safe = build_single_student_attempt_figure(comparison, "S1", "Grade", "Improved", colorblind_mode=True)
    assert normal.data[0].line.color != safe.data[0].line.color
