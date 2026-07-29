from analytics.expression_tree import parse_expression


def test_flattens_multiplication_chain():
    node = parse_expression("2*x*sqrt(x)")
    assert node.label == "*"
    assert [c.label for c in node.children] == ["2", "x", "sqrt"]
    assert node.children[2].children[0].label == "x"


def test_unary_minus_on_literal_folds_into_atom():
    node = parse_expression("-1/(2*x*sqrt(x))")
    assert node.label == "/"
    assert node.children[0].label == "-1"
    assert node.children[0].children == []


def test_unary_minus_on_non_literal_is_a_node():
    node = parse_expression("-x")
    assert node.label == "-"
    assert len(node.children) == 1
    assert node.children[0].label == "x"


def test_function_call_children_are_arguments():
    node = parse_expression("nthroot(x, 3)")
    assert node.label == "nthroot"
    assert [c.label for c in node.children] == ["x", "3"]


def test_addition_chain_flattens_but_mixed_chain_nests():
    plus_chain = parse_expression("a+b+c")
    assert plus_chain.label == "+"
    assert [c.label for c in plus_chain.children] == ["a", "b", "c"]

    mixed_chain = parse_expression("a-b+c")
    assert mixed_chain.label == "+"
    assert mixed_chain.children[0].label == "-"
    assert [c.label for c in mixed_chain.children[0].children] == ["a", "b"]
    assert mixed_chain.children[1].label == "c"


def test_power_is_right_associative():
    node = parse_expression("2^3^2")
    assert node.label == "^"
    assert node.children[0].label == "2"
    assert node.children[1].label == "^"


def test_unparseable_or_empty_expression_returns_none():
    assert parse_expression("matrix([1,2],[3,4])") is None
    assert parse_expression("") is None
    assert parse_expression(None) is None
    assert parse_expression("1 + + 2") is None
