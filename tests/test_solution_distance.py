import pandas as pd

from analytics.parser import parse_response_cell
from analytics.solution_distance import (
    BACKDROP_TRACE_NAME,
    _build_distance_colorscale,
    build_prt_distance_3d_figure,
    build_ted_distance_3d_figure,
    compute_prt_distance_series,
    compute_question_student_order,
    compute_ted_distance_series,
)

RIGHT_ANSWER = "ans1: -1/(2*x*sqrt(x)) [valid]"

CORRECT = "Seed: 1; ans1: -1/(2*x*sqrt(x)) [score]; prt1: # = 1 | prt1-1-T"
NODE2_TRUE = "Seed: 1; ans1: 1/(2*sqrt(x)) [score]; prt1: # = 0.5 | prt1-1-F | prt1-2-T"
NODE3_TRUE = "Seed: 1; ans1: x^(-3/2) [score]; prt1: # = 0.1 | prt1-1-F | prt1-2-F | prt1-3-T"
ALL_FALSE = "Seed: 1; ans1: 7 [score]; prt1: # = 0 | prt1-1-F | prt1-2-F | prt1-3-F"


def _student_traces(fig):
    """The per-student trajectory traces, without the scene's walls and gridlines."""
    return [trace for trace in fig.data if trace.name != BACKDROP_TRACE_NAME]


def _row(student_id, attempt_idx, cell, question="Q1", right_answer=RIGHT_ANSWER):
    ans_list, prt_list = parse_response_cell(cell)
    return {
        "question": question,
        "student_id": student_id,
        "student_name": student_id.upper(),
        "response_status": "incorrect",
        "prt_list": prt_list,
        "ans_list": ans_list,
        "right_answer_text": right_answer,
        "grade": 0.0,
        "overall_grade": 0.0,
        "completed_dt": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=attempt_idx),
        "attempt_idx": attempt_idx,
    }


def test_prt_distance_matches_table1_style_generalization():
    """Takada et al. Table 1: node 1 True = correct (0); node 2 True = 1; node 3 True = 2;
    matching nothing lands in the shared "other" sentinel bucket."""
    df = pd.DataFrame([
        _row("s1", 1, CORRECT),
        _row("s2", 1, NODE2_TRUE),
        _row("s3", 1, NODE3_TRUE),
        _row("s4", 1, ALL_FALSE),
    ])
    out = compute_prt_distance_series(df, "Q1")
    dist = dict(zip(out["student_id"], out["prt_distance"]))
    assert dist["s1"] == 0
    assert dist["s2"] == 1
    assert dist["s3"] == 2
    assert dist["s4"] == 2 + 3  # sentinel = max classified distance (2) + 3


def test_prt_distance_is_per_part():
    cell = ("Seed: 1; ans1: a [score]; ans2: b [score]; "
            "prt1: # = 1 | prt1-1-T; prt2: # = 0.5 | prt2-1-F | prt2-2-T")
    df = pd.DataFrame([_row("s1", 1, cell)])
    assert compute_prt_distance_series(df, "Q1", part_index=1).iloc[0]["prt_distance"] == 0
    assert compute_prt_distance_series(df, "Q1", part_index=2).iloc[0]["prt_distance"] == 1


def test_student_order_is_a_nested_sort_over_every_attempt():
    """Primary grouping by the first attempt, then the second attempt breaks ties inside
    each of those clusters, and so on recursively."""
    df = pd.DataFrame([
        _row("zed", 1, CORRECT),                             # (0,)
        _row("amy", 1, CORRECT),                             # (0,)
        _row("bob", 1, NODE2_TRUE), _row("bob", 2, NODE3_TRUE),  # (1, 2)
        _row("cat", 1, NODE2_TRUE), _row("cat", 2, CORRECT),     # (1, 0)
        _row("dan", 1, NODE2_TRUE), _row("dan", 2, NODE2_TRUE),  # (1, 1)
        _row("eve", 1, NODE3_TRUE), _row("eve", 2, CORRECT),     # (2, 0)
    ])
    subset = compute_prt_distance_series(df, "Q1")
    order = compute_question_student_order(subset, "prt_distance")
    ranked = sorted(order, key=lambda sid: order[sid])
    # correct-on-first-try cluster first (alphabetical within it), then the distance-1
    # cluster ordered by second attempt, then the distance-2 cluster.
    assert ranked == ["amy", "zed", "cat", "dan", "bob", "eve"]


def test_student_order_prefers_shorter_run_on_an_equal_prefix():
    df = pd.DataFrame([
        _row("s1", 1, NODE2_TRUE), _row("s1", 2, CORRECT),                        # (1, 0)
        _row("s2", 1, NODE2_TRUE), _row("s2", 2, CORRECT), _row("s2", 3, ALL_FALSE),  # (1, 0, 5)
    ])
    subset = compute_prt_distance_series(df, "Q1")
    order = compute_question_student_order(subset, "prt_distance")
    assert order["s1"] < order["s2"]


def test_build_distance_colorscale_anchors():
    scale = _build_distance_colorscale(10)
    assert scale[0] == [0.0, "#FFFFFF"]   # distance 0 is white
    assert scale[1] == [1.0 / 10, "#FF1744"]  # distance 1 is neon red
    assert scale[-1] == [1.0, "#000000"]
    positions = [stop[0] for stop in scale]
    assert positions == sorted(positions) and len(positions) == len(set(positions))


def test_build_distance_colorscale_handles_small_max_without_duplicate_stops():
    # max_value <= 1 leaves no room for the remaining anchors — must not emit duplicate
    # stop positions (which Plotly's colorscale validation rejects).
    for max_value in (0, 1):
        scale = _build_distance_colorscale(max_value)
        positions = [stop[0] for stop in scale]
        assert positions == sorted(positions) and len(positions) == len(set(positions))


def test_build_distance_colorscale_stops_never_exceed_one():
    # Regression test: `span * index / last_index` previously accumulated floating-point
    # error and landed the final stop a hair above 1.0 (e.g. 1.0000000000000002), which
    # Plotly's colorscale validator rejects outright since positions must stay in [0, 1].
    for max_value in (3, 7, 9, 10, 13, 20):
        scale = _build_distance_colorscale(max_value)
        assert scale[-1][0] == 1.0
        assert all(0.0 <= stop[0] <= 1.0 for stop in scale)


def test_ted_distance_series_matches_figure_2_example():
    df = pd.DataFrame([_row("s1", 1, NODE2_TRUE)])
    out = compute_ted_distance_series(df, "Q1")
    assert out.iloc[0]["ted_distance"] == 2


def test_ted_distance_none_when_unparseable():
    df = pd.DataFrame([
        _row("s1", 1, "Seed: 1; ans1: matrix([1,2]) [score]; prt1: # = 0 | prt1-1-F"),
    ])
    out = compute_ted_distance_series(df, "Q1")
    assert out.iloc[0]["ted_distance"] is None


def test_ted_distance_uses_the_selected_parts_expression():
    cell = ("Seed: 1; ans1: -1/(2*x*sqrt(x)) [score]; ans2: 1/(2*sqrt(x)) [score]; "
            "prt1: # = 1 | prt1-1-T; prt2: # = 0 | prt2-1-F")
    right = "ans1: -1/(2*x*sqrt(x)) [valid]; ans2: -1/(2*x*sqrt(x)) [valid]"
    df = pd.DataFrame([_row("s1", 1, cell, right_answer=right)])
    assert compute_ted_distance_series(df, "Q1", part_index=1).iloc[0]["ted_distance"] == 0
    assert compute_ted_distance_series(df, "Q1", part_index=2).iloc[0]["ted_distance"] == 2


def test_first_attempt_correct_student_is_plotted_at_the_origin():
    """A student who was right first time has a single point at attempt 0 / distance 0 —
    they must still be drawn (as the white dot nearest the origin), not dropped."""
    df = pd.DataFrame([
        _row("s1", 1, CORRECT),
        _row("s2", 1, NODE2_TRUE), _row("s2", 2, CORRECT),
    ])
    for fig in (build_prt_distance_3d_figure(df, "Q1"), build_ted_distance_3d_figure(df, "Q1")):
        traces = {trace.text[0]: trace for trace in _student_traces(fig)}
        assert "S1" in traces, "first-attempt-correct student missing from the chart"
        s1 = traces["S1"]
        # Attempt is on y, student rank on x — see `_build_distance_3d_figure`.
        assert list(s1.y) == [0] and list(s1.z) == [0]
        assert list(s1.x) == [1], "should sort closest to the origin on the Students axis"
        # White-on-white would be invisible, so the marker carries an outline.
        assert s1.marker.line.width == 1


def test_build_3d_figures_smoke():
    df = pd.DataFrame([_row("s1", 1, NODE2_TRUE), _row("s1", 2, CORRECT)])
    prt_fig = build_prt_distance_3d_figure(df, "Q1")
    ted_fig = build_ted_distance_3d_figure(df, "Q1")
    for fig in (prt_fig, ted_fig):
        assert len(_student_traces(fig)) > 0
        # Attempt counts and distances are both whole numbers, so no axis may show
        # fractional ticks ("attempt 0.5").
        for axis in (fig.layout.scene.xaxis, fig.layout.scene.yaxis, fig.layout.scene.zaxis):
            assert axis.tickvals, "ticks must be pinned, not left to autoscale"
            assert all(float(tick).is_integer() for tick in axis.tickvals)


def test_scene_backdrop_is_fixed_geometry_not_flipping_panes():
    """The walls have to be traces pinned to fixed planes; Plotly's own panes are redrawn on
    whichever faces face away from the camera, so they swap sides mid-drag.

    Each wall belongs on its axis's *far* face, which is always the first value of that
    axis's range: for an ascending axis that is the low end, and for the reversed Students
    axis it is the high end. Their other two extents must span the full range, or the box
    won't close."""
    df = pd.DataFrame([_row("s1", 1, NODE2_TRUE), _row("s1", 2, CORRECT)])
    fig = build_prt_distance_3d_figure(df, "Q1")
    scene = fig.layout.scene

    for axis in (scene.xaxis, scene.yaxis, scene.zaxis):
        assert axis.showbackground is False
        assert axis.showgrid is False

    walls = [t for t in fig.data if t.name == BACKDROP_TRACE_NAME and t.type == "mesh3d"]
    assert len(walls) == 3, "expected a floor and two side walls"
    assert all(wall.opacity is None for wall in walls), (
        "walls must stay opaque — Plotly composites translucent meshes over the whole "
        "scene, washing the data out"
    )
    ranges = {"x": scene.xaxis.range, "y": scene.yaxis.range, "z": scene.zaxis.range}
    pinned = set()
    for wall in walls:
        coords = {"x": wall.x, "y": wall.y, "z": wall.z}
        flat = [name for name, values in coords.items() if len(set(values)) == 1]
        assert len(flat) == 1, "each wall lies in one plane"
        held = flat[0]
        far_end, near_end = ranges[held]
        # Pinned to the far face — the three that sit behind the data at the default eye.
        assert coords[held][0] == far_end, f"{held} wall must sit at {far_end}, not {near_end}"
        pinned.add(held)
        for name in coords.keys() - {held}:
            assert (min(coords[name]), max(coords[name])) == tuple(sorted(ranges[name])), (
                f"wall must span the full {name} range so the box closes"
            )
    assert pinned == {"x", "y", "z"}, "one wall per plane: floor, back and side"


def test_students_axis_runs_back_to_front_without_moving_any_student():
    """Rank 1 must sit nearest the viewer. The flip has to be a coordinate-space flip, not
    relabelled ticks, so every student's own x/y/z stays exactly as it was — and the side
    wall has to move to the axis's new far face, or it ends up in front of the data."""
    df = pd.DataFrame([
        _row("s1", 1, CORRECT),
        _row("s2", 1, NODE2_TRUE), _row("s2", 2, CORRECT),
        _row("s3", 1, ALL_FALSE), _row("s3", 2, NODE3_TRUE), _row("s3", 3, CORRECT),
    ])
    for build in (build_prt_distance_3d_figure, build_ted_distance_3d_figure):
        fig = build(df, "Q1")
        low, high = sorted(fig.layout.scene.xaxis.range)
        assert list(fig.layout.scene.xaxis.range) == [high, low], "Students axis must be reversed"

        # Ranks are still 1..N on the data itself, one constant x per student.
        ranks = sorted(trace.x[0] for trace in _student_traces(fig))
        assert ranks == list(range(1, len(ranks) + 1))
        for trace in _student_traces(fig):
            assert len(set(trace.x)) == 1, "a student sits at one rank across their attempts"
            assert list(trace.y) == list(range(len(trace.y))), "attempts stay 0..n-1"

        walls = [t for t in fig.data if t.name == BACKDROP_TRACE_NAME and t.type == "mesh3d"]
        x_wall = next(w.x[0] for w in walls if len(set(w.x)) == 1)
        assert x_wall == high, "the side wall belongs on the reversed axis's far face"


def test_both_3d_charts_open_from_the_same_fixed_viewpoint():
    df = pd.DataFrame([_row("s1", 1, NODE2_TRUE), _row("s1", 2, CORRECT)])
    prt_fig, ted_fig = build_prt_distance_3d_figure(df, "Q1"), build_ted_distance_3d_figure(df, "Q1")
    prt_eye, ted_eye = prt_fig.layout.scene.camera.eye, ted_fig.layout.scene.camera.eye
    assert (prt_eye.x, prt_eye.y, prt_eye.z) == (ted_eye.x, ted_eye.y, ted_eye.z)
    # Positive on all three: looking into the box from the +x/+y/+z corner. At that eye
    # the x axis recedes to the right and y to the left, so plotting students on x and
    # attempts on y is what puts Students down-right and Attempt down-left on screen.
    assert prt_eye.x > 0 and prt_eye.y > 0 and prt_eye.z > 0
    for fig in (prt_fig, ted_fig):
        assert fig.layout.scene.xaxis.title.text == "Students"
        assert fig.layout.scene.yaxis.title.text == "Attempt"
