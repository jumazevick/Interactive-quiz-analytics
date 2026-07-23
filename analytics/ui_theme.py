from __future__ import annotations

import pandas as pd
import streamlit as st

_GLOBAL_CSS = """
<style>
/* Buttons — consistent rounded, subtle-lift styling across st.button /
   st.download_button / st.link_button, all of which have stable, documented testids. */
div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button,
div[data-testid="stLinkButton"] a {
    border-radius: 8px;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover,
div[data-testid="stLinkButton"] a:hover {
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.15);
}

/* Metrics — bolder values read better as at-a-glance stats. */
div[data-testid="stMetricValue"] {
    font-weight: 700;
}

/* Headings — slightly tighter tracking reads a touch more modern. */
h1, h2, h3 {
    letter-spacing: -0.01em;
}

/* Dividers — softer than the browser default, matches the muted footer text. */
hr {
    border-color: rgba(128, 128, 128, 0.25);
}
</style>
"""


def inject_global_styles() -> None:
    """Shared, presentation-only CSS polish applied on every page. Deliberately limited
    to plain HTML tags and Streamlit's officially documented component testids (stButton,
    stDownloadButton, stLinkButton, stMetricValue) rather than internal emotion-cache
    classes, which are undocumented and can silently break on a Streamlit version bump.
    """
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


_ACRONYM_WORDS = {"id", "prt"}


def humanize_column_name(name: str) -> str:
    """snake_case metric name -> Title Case column header (e.g. `average_marks` ->
    `Average Marks`, `student_id` -> `Student ID`). Idempotent on names that are
    already human-readable (e.g. `Student Name` passes through unchanged), so it's
    safe to apply to a DataFrame whose columns are a mix of raw metric keys and
    already-labeled columns."""
    words = str(name).replace("_", " ").split()
    return " ".join(w.upper() if w.lower() in _ACRONYM_WORDS else w.capitalize() for w in words)


def humanize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with every column renamed via `humanize_column_name`, so
    tables read consistently (e.g. "Average Marks" instead of "average_marks") whether
    they're shown with st.dataframe or embedded in the PDF export — both consume the
    DataFrame's columns directly as headers."""
    return df.rename(columns={col: humanize_column_name(col) for col in df.columns})
