from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

# Okabe-Ito colorblind-safe qualitative palette (px.colors.qualitative.Safe), used in
# place of Set1/Set2/Vivid/Bold for every categorical chart when colorblind mode is on.
COLORBLIND_QUALITATIVE_PALETTE = px.colors.qualitative.Safe

# Replaces the red/yellow/green pass-rate heatmap scale — red vs. green is the one
# combination red-green colorblind users can't reliably tell apart, which is exactly
# what that scale uses to encode pass/fail. Blue/yellow/vermillion (Okabe-Ito) reads
# clearly for both red-green (deuteranopia/protanopia) and typical vision.
COLORBLIND_PASS_FAIL_SCALE = ["#0072B2", "#F0E442", "#D55E00"]
DEFAULT_PASS_FAIL_SCALE = ["#ef4444", "#fde68a", "#22c55e"]


def qualitative_colors(colorblind_mode: bool, default: list[str]) -> list[str]:
    """Categorical chart palette: `default` normally, or the colorblind-safe Safe
    palette when colorblind_mode is on."""
    return COLORBLIND_QUALITATIVE_PALETTE if colorblind_mode else default


def pass_fail_scale(colorblind_mode: bool) -> list[str]:
    """Diverging red/yellow/green pass-rate scale, or its colorblind-safe equivalent."""
    return COLORBLIND_PASS_FAIL_SCALE if colorblind_mode else DEFAULT_PASS_FAIL_SCALE


_GLOBAL_CSS = """
<style>
/* Sidebar — wider default so longer checkbox/button labels (e.g. "5. Student
   Performance by Question", "Clear / Reset All Uploaded Files") fit on one line
   instead of wrapping. Users can still drag-resize it narrower/wider afterward.
   Scoped to aria-expanded="true": Streamlit collapses the sidebar by animating its
   inline `width` down to a small value, and CSS min-width always wins over a smaller
   inline width — so an unscoped min-width here would stop the sidebar from ever
   fully collapsing (it'd get stuck open by the difference, arrow and all). */
section[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 370px !important;
    /* Streamlit animates min-width/max-width/transform over 0.3s for its own
       collapse/resize UX. Between multipage navigations, the sidebar briefly renders
       at Streamlit's native default width before this stylesheet re-mounts, and
       without this override that correction visibly slides in — a "pop" — instead of
       just snapping to the right width. transform stays animated so manual
       collapse/expand still slides smoothly; only the width snap is instant. */
    transition: transform 0.3s !important;
}

/* Sidebar collapse arrow — Streamlit hides this (visibility: hidden) until the
   sidebar is hovered, which makes it easy to lose track of how to collapse the
   sidebar again. Keep it always visible while the sidebar is expanded. */
div[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
}

/* Sidebar scrolling — stSidebarContent already scrolls as ONE region covering the
   page nav links and everything below (uploader, checkboxes, etc.), which is what we
   want. But Streamlit also gives the nested stSidebarUserContent div its own
   independent overflow, so the lower portion of the sidebar gets a second, separate
   scrollbar that scrolls out of sync with the first — that's the "two scroll things"
   bug. Disabling just that inner one leaves stSidebarContent as the single scroll
   region for the whole sidebar. */
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
    overflow-y: visible !important;
}

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
    letter-spacing: -0.015em;
    font-weight: 700;
}

/* Dividers — softer than the browser default, matches the muted footer text. */
hr {
    border-color: rgba(128, 128, 128, 0.25);
}

/* Sidebar page nav — subtle rounded hover state instead of the flat default rows,
   matching the rounded, bordered look used for buttons/containers elsewhere. */
div[data-testid="stSidebarNav"] a {
    border-radius: 8px;
    transition: background-color 0.15s ease;
}
div[data-testid="stSidebarNav"] a:hover {
    background-color: rgba(128, 128, 128, 0.12);
}

/* Checkboxes — a touch more breathing room between stacked toggles in the sidebar,
   and a subtle hover cue; the checked-state color itself comes from the active
   theme's primaryColor (see .streamlit/config.toml) rather than being hardcoded here. */
div[data-testid="stCheckbox"] {
    padding: 0.1rem 0;
}
div[data-testid="stCheckbox"] label:hover {
    opacity: 0.85;
}
/* The visual box+checkmark is the label's other direct-child div (its sibling is the
   text container, stWidgetLabel — a documented testid we anchor on instead of any
   internal class name). Scaling it up ~15% makes it easier to spot/click without
   changing the label font size or row height around it. */
div[data-testid="stCheckbox"] label > div:not([data-testid="stWidgetLabel"]) {
    transform: scale(1.15);
    transform-origin: center;
}
</style>
"""


def inject_global_styles() -> None:
    """Shared, presentation-only CSS polish applied on every page. Deliberately limited
    to plain HTML tags and Streamlit's officially documented component testids (stButton,
    stDownloadButton, stLinkButton, stMetricValue, stSidebarNav, stCheckbox) rather than
    internal emotion-cache classes, which are undocumented and can silently break on a
    Streamlit version bump. Color choices themselves live in .streamlit/config.toml so
    both the dark and light theme variants (toggled from the app's built-in menu) stay
    in sync automatically.
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
