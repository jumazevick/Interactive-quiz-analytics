from analytics.expression_tree import parse_expression
from analytics.tree_edit_distance import tree_edit_distance


def test_figure_2_worked_example():
    """Takada et al. (EDM 2025) Figure 2: 1/(2*sqrt(x)) -> -1/(2*x*sqrt(x)) via one
    rename (1 -> -1) and one insert (x into the * node) — TED must equal 2."""
    a = parse_expression("1/(2*sqrt(x))")
    b = parse_expression("-1/(2*x*sqrt(x))")
    assert tree_edit_distance(a, b) == 2


def test_identical_trees_have_zero_distance():
    a = parse_expression("x^2 + 1")
    b = parse_expression("x^2 + 1")
    assert tree_edit_distance(a, b) == 0


def test_distance_is_symmetric():
    a = parse_expression("sin(x) + cos(x)")
    b = parse_expression("cos(x) + sin(x) + 1")
    assert tree_edit_distance(a, b) == tree_edit_distance(b, a)


def test_single_rename_costs_one():
    a = parse_expression("x + 1")
    b = parse_expression("y + 1")
    assert tree_edit_distance(a, b) == 1
