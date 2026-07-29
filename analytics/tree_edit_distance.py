from __future__ import annotations

from analytics.expression_tree import ExprNode


def _postorder(root: ExprNode) -> list[ExprNode]:
    """Postorder traversal (children before parent), 0-indexed list where index `i`
    corresponds to 1-indexed postorder position `i + 1` in the classic Zhang-Shasha
    formulation used below."""
    order: list[ExprNode] = []

    def _walk(node: ExprNode) -> None:
        for child in node.children:
            _walk(child)
        order.append(node)

    _walk(root)
    return order


def _leftmost_leaf_positions(nodes: list[ExprNode]) -> list[int]:
    """1-indexed `l(i)`: the postorder position of node `i`'s leftmost leaf descendant
    (itself, if `i` is a leaf). `l[0]` is an unused placeholder so `l[i]` lines up with
    1-indexed postorder position `i`."""
    n = len(nodes)
    position_of_id = {id(node): index + 1 for index, node in enumerate(nodes)}
    l = [0] * (n + 1)
    for i in range(1, n + 1):
        node = nodes[i - 1]
        if not node.children:
            l[i] = i
        else:
            first_child_pos = position_of_id[id(node.children[0])]
            l[i] = l[first_child_pos]
    return l


def _keyroots(l: list[int], n: int) -> list[int]:
    """Keyroots: for each distinct `l(i)` value, the largest postorder index sharing it
    — equivalent to (and simpler to compute than) "root, or l(i) != l(parent(i))"."""
    last_index_for_l: dict[int, int] = {}
    for i in range(1, n + 1):
        last_index_for_l[l[i]] = i
    return sorted(last_index_for_l.values())


def tree_edit_distance(tree_a: ExprNode, tree_b: ExprNode) -> int:
    """Zhang-Shasha edit distance between two ordered labeled trees (Zhang & Shasha,
    1989), where insert/delete/rename each cost 1 — the edit-operation model Takada et
    al. (EDM 2025) use to measure how far a student's submitted CAS expression tree is
    from the correct answer's tree."""
    nodes_a = _postorder(tree_a)
    nodes_b = _postorder(tree_b)
    n, m = len(nodes_a), len(nodes_b)

    l_a = _leftmost_leaf_positions(nodes_a)
    l_b = _leftmost_leaf_positions(nodes_b)
    keyroots_a = _keyroots(l_a, n)
    keyroots_b = _keyroots(l_b, m)

    label_a = {i: nodes_a[i - 1].label for i in range(1, n + 1)}
    label_b = {i: nodes_b[i - 1].label for i in range(1, m + 1)}

    treedist: dict[tuple[int, int], int] = {}

    for i in keyroots_a:
        for j in keyroots_b:
            forestdist: dict[tuple[int, int], int] = {(l_a[i] - 1, l_b[j] - 1): 0}

            for i1 in range(l_a[i], i + 1):
                forestdist[(i1, l_b[j] - 1)] = forestdist[(i1 - 1, l_b[j] - 1)] + 1

            for j1 in range(l_b[j], j + 1):
                forestdist[(l_a[i] - 1, j1)] = forestdist[(l_a[i] - 1, j1 - 1)] + 1

            for i1 in range(l_a[i], i + 1):
                for j1 in range(l_b[j], j + 1):
                    if l_a[i1] == l_a[i] and l_b[j1] == l_b[j]:
                        rename_cost = 0 if label_a[i1] == label_b[j1] else 1
                        dist = min(
                            forestdist[(i1 - 1, j1)] + 1,
                            forestdist[(i1, j1 - 1)] + 1,
                            forestdist[(i1 - 1, j1 - 1)] + rename_cost,
                        )
                        forestdist[(i1, j1)] = dist
                        treedist[(i1, j1)] = dist
                    else:
                        dist = min(
                            forestdist[(i1 - 1, j1)] + 1,
                            forestdist[(i1, j1 - 1)] + 1,
                            forestdist[(l_a[i1] - 1, l_b[j1] - 1)] + treedist[(i1, j1)],
                        )
                        forestdist[(i1, j1)] = dist

    return treedist[(n, m)]
