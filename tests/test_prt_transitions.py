from collections import Counter

import pandas as pd

from analytics.parser import parse_response_cell
from analytics.prt_transitions import (
    build_aggregate_graph,
    build_student_node_sequence,
    build_transition_graph_figure,
    build_transition_pairs,
    classify_node,
    compute_network_features,
    count_question_parts,
    node_sort_key,
)

# Realistic STACK response cells, parsed through the real parser so these tests cover
# the parser -> classifier path end to end. Node labels follow the reference
# implementation (network_analysis.py): node 1 True = correct, node k True -> k-1,
# nothing True = unclassified.
CORRECT = "Seed: 1; ans1: x [score]; prt1: # = 1 | prt1-1-T"
NODE2_TRUE = "Seed: 1; ans1: x [score]; prt1: # = 0.5 | prt1-1-F | prt1-2-T"
NODE3_TRUE = "Seed: 1; ans1: x [score]; prt1: # = 0.1 | prt1-1-F | prt1-2-F | prt1-3-T"
ALL_FALSE = "Seed: 1; ans1: x [score]; prt1: # = 0 | prt1-1-F | prt1-2-F | prt1-3-F"
BLANK = "Seed: 1"


def _row(student_id, attempt_idx, cell, question="Q1"):
    ans_list, prt_list = parse_response_cell(cell)
    return {
        "question": question,
        "student_id": student_id,
        "student_name": student_id.upper(),
        "response_status": "incorrect",
        "prt_list": prt_list,
        "ans_list": ans_list,
        "grade": 0.0,
        "overall_grade": 0.0,
        "completed_dt": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=attempt_idx),
        "attempt_idx": attempt_idx,
    }


def _series(cell):
    _, prt_list = parse_response_cell(cell)
    return pd.Series({"response_status": "incorrect", "prt_list": prt_list})


def test_classify_node_matches_reference_script_mapping():
    """network_analysis.py derives answer_number from whichever node evaluated True and
    applies mapping = {0: 0, 1: 'c', 2: 1, 3: 2, ...}."""
    assert classify_node(_series(CORRECT)) == "c"      # node 1 True
    assert classify_node(_series(NODE2_TRUE)) == "1"   # node 2 True -> 1
    assert classify_node(_series(NODE3_TRUE)) == "2"   # node 3 True -> 2


def test_classify_node_treats_full_fallthrough_as_unclassified():
    """A trace ending '-F' matched no node at all, so it is an unclassified wrong answer
    ("0") — not node 3. Reading only the terminal note's number invented a category the
    PRT never assigned."""
    assert classify_node(_series(ALL_FALSE)) == "0"


def test_classify_node_blank_response_is_unclassified():
    assert classify_node(_series(BLANK)) == "0"


def test_classify_node_is_per_part_on_multi_part_questions():
    cell = (
        "Seed: 1; ans1: a [score]; ans2: b [score]; ans3: c [score]; "
        "prt1: # = 1 | prt1-1-T; "
        "prt2: # = 0.5 | prt2-1-F | prt2-2-T; "
        "prt3: # = 0 | prt3-1-F | prt3-2-F | prt3-3-F"
    )
    row = _series(cell)
    assert classify_node(row, part_index=1) == "c"
    assert classify_node(row, part_index=2) == "1"
    assert classify_node(row, part_index=3) == "0"
    # A part index the question doesn't have is unclassified rather than an error.
    assert classify_node(row, part_index=9) == "0"


def test_classify_node_uses_per_part_marks_not_whole_question_status():
    """On a multi-part question response_status is only "correct" when *every* part is
    right, so it can't stand in for "this part was right"."""
    cell = ("Seed: 1; ans1: a [score]; ans2: b [score]; "
            "prt1: # = 1 | prt1-1-T; prt2: # = 0 | prt2-1-F")
    _, prt_list = parse_response_cell(cell)
    row = pd.Series({"response_status": "incorrect", "prt_list": prt_list})
    assert classify_node(row, part_index=1) == "c"
    assert classify_node(row, part_index=2) == "0"


def test_count_question_parts():
    cell = ("Seed: 1; ans1: a [score]; ans2: b [score]; ans3: c [score]; "
            "prt1: # = 1 | prt1-1-T; prt2: # = 1 | prt2-1-T; prt3: # = 1 | prt3-1-T")
    df = pd.DataFrame([_row("s1", 1, cell), _row("s1", 2, CORRECT)])
    assert count_question_parts(df, "Q1") == 3
    # Single-part question still reports 1 rather than 0.
    assert count_question_parts(pd.DataFrame([_row("s1", 1, CORRECT)]), "Q1") == 1


def test_node_sort_key_orders_0_then_numeric_then_c():
    assert sorted(["c", "3", "0", "1"], key=node_sort_key) == ["0", "1", "3", "c"]


def test_student_sequence_and_transition_pairs():
    df = pd.DataFrame([
        _row("s1", 1, ALL_FALSE),    # 0
        _row("s1", 2, NODE3_TRUE),   # 2
        _row("s1", 3, CORRECT),      # c
    ])
    seq = build_student_node_sequence(df, "Q1", "s1")
    assert seq["node"].tolist() == ["0", "2", "c"]
    assert build_transition_pairs(seq["node"].tolist()) == [("0", "2"), ("2", "c")]


def test_aggregate_graph_includes_self_loop_and_weights():
    df = pd.DataFrame([
        _row("s1", 1, NODE2_TRUE),   # 1
        _row("s1", 2, NODE2_TRUE),   # 1  -> self-loop
        _row("s1", 3, CORRECT),      # c
        _row("s2", 1, NODE2_TRUE),   # 1
        _row("s2", 2, CORRECT),      # c
    ])
    nodes, edges = build_aggregate_graph(df, "Q1")
    assert nodes == ["0", "1", "c"]
    assert edges[("1", "1")] == 1
    assert edges[("1", "c")] == 2


def test_compute_network_features_degree_and_centrality():
    nodes = ["0", "2", "c"]
    edges = Counter({("2", "2"): 1, ("2", "c"): 2})
    features = compute_network_features(nodes, edges).set_index("node")

    assert features.loc["0", "in_degree"] == 0
    assert features.loc["0", "out_degree"] == 0

    assert features.loc["2", "in_degree"] == 1   # from its own self-loop
    assert features.loc["2", "out_degree"] == 3  # self-loop (1) + to "c" (2)
    assert features.loc["2", "degree"] == 4
    assert features.loc["2", "in_degree_centrality"] == 0.5   # denom = n - 1 = 2
    assert features.loc["2", "out_degree_centrality"] == 1.5

    assert features.loc["c", "in_degree"] == 2
    assert features.loc["c", "out_degree"] == 0


def test_network_features_always_include_the_correct_node():
    """"c" must appear as its own row so the centrality bar charts can plot it."""
    nodes, edges = build_aggregate_graph(
        pd.DataFrame([_row("s1", 1, NODE2_TRUE), _row("s1", 2, CORRECT)]), "Q1"
    )
    features = compute_network_features(nodes, edges)
    assert "c" in features["node"].tolist()


def test_build_transition_graph_figure_smoke():
    nodes = ["0", "2", "c"]
    edges = Counter({("2", "2"): 1, ("2", "c"): 2})
    assert len(build_transition_graph_figure(nodes, edges, colorblind_mode=False, title="T").data) > 0
    assert len(build_transition_graph_figure(nodes, edges, colorblind_mode=True, title="T").data) > 0
