import pandas as pd

from analytics.quiz_metrics import build_scatter_figure


def _attempt_frame(rows: list[tuple[str, str, int, float]]) -> pd.DataFrame:
    """rows of (quiz_name, student_id, attempts, grade) -> one row per attempt, matching
    the real attempt_frame shape build_scatter_figure expects (grouped by size() to
    recover attempt count)."""
    out = []
    for quiz_name, student_id, attempts, grade in rows:
        for _ in range(attempts):
            out.append({"quiz_name": quiz_name, "student_id": student_id, "overall_grade": grade})
    return pd.DataFrame(out)


def test_correlation_is_computed_from_true_values_not_jittered_ones():
    """The reported r must describe the real data; jitter is display-only."""
    df = _attempt_frame([
        ("Quiz A", "s1", 1, 10.0), ("Quiz A", "s2", 2, 8.0),
        ("Quiz A", "s3", 3, 6.0), ("Quiz A", "s4", 4, 4.0),
    ])
    fig, correlation, y_label, title = build_scatter_figure(df, "Average Grade")
    expected = pd.Series([1, 2, 3, 4]).corr(pd.Series([10.0, 8.0, 6.0, 4.0]))
    assert abs(correlation - expected) < 1e-9


def test_jitter_is_deterministic_across_rebuilds():
    """The same input must produce the same plotted positions every time -- jitter that
    reshuffled on rerun would be worse than the overlap it exists to fix."""
    df = _attempt_frame([
        ("Quiz A", "s1", 1, 10.0), ("Quiz B", "s2", 1, 10.0), ("Quiz C", "s3", 1, 10.0),
    ])
    fig1, *_ = build_scatter_figure(df, "Average Grade")
    fig2, *_ = build_scatter_figure(df, "Average Grade")
    for t1, t2 in zip(fig1.data, fig2.data):
        assert list(t1.x) == list(t2.x)
        assert list(t1.y) == list(t2.y)


def test_jitter_stays_within_the_target_amplitude():
    """The true integer attempt count must still be visually obvious -- offsets bounded
    to +/-0.15 of it, per the target amplitude."""
    df = _attempt_frame([(f"Quiz {q}", f"s{i}", 1, 10.0) for q in "ABCD" for i in range(20)])
    fig, *_ = build_scatter_figure(df, "Average Grade")
    for trace in fig.data:
        true_x = trace.customdata[:, 0]
        for plotted, true in zip(trace.x, true_x):
            assert abs(plotted - true) <= 0.15 + 1e-9


def test_overlapping_quizzes_are_not_collapsed_into_one_trace():
    """Every quiz keeps its own trace/legend entry -- jitter must not merge series."""
    df = _attempt_frame([(f"Quiz {q}", f"s{q}", 1, 10.0) for q in "ABCD"])
    fig, *_ = build_scatter_figure(df, "Average Grade")
    assert {t.name for t in fig.data} == {"Quiz A", "Quiz B", "Quiz C", "Quiz D"}


def test_hover_shows_true_coordinates_not_jittered_ones():
    df = _attempt_frame([("Quiz A", "s1", 3, 7.5)])
    fig, *_ = build_scatter_figure(df, "Average Grade")
    trace = fig.data[0]
    assert trace.customdata[0][0] == 3
    assert trace.customdata[0][1] == 7.5
    assert "customdata[0]" in trace.hovertemplate
    assert "customdata[1]" in trace.hovertemplate


def test_marker_size_grows_with_shared_coordinate_density():
    """A coordinate ten students share should render larger than one nobody else is on,
    as a density cue that doesn't rely on jitter/opacity alone."""
    rows = [("Quiz A", f"crowded{i}", 1, 10.0) for i in range(10)]
    rows.append(("Quiz A", "lonely", 5, 3.0))
    df = _attempt_frame(rows)
    fig, *_ = build_scatter_figure(df, "Average Grade")
    sizes = list(fig.data[0].marker.size)
    customdata = fig.data[0].customdata
    crowded_sizes = [s for s, c in zip(sizes, customdata) if c[0] == 1]
    lonely_size = next(s for s, c in zip(sizes, customdata) if c[0] == 5)
    assert min(crowded_sizes) > lonely_size


def test_x_axis_ticks_stay_at_integers():
    df = _attempt_frame([("Quiz A", "s1", 1, 10.0), ("Quiz A", "s2", 3, 5.0)])
    fig, *_ = build_scatter_figure(df, "Average Grade")
    assert fig.layout.xaxis.dtick == 1


def test_empty_attempt_frame_returns_none():
    assert build_scatter_figure(pd.DataFrame(), "Average Grade") is None
