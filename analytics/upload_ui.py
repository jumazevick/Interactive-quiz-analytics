from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.anonymize import anonymize_response_df
from analytics.data_loader import load_quiz_data
from analytics.upload_cache import (
    ANONYMIZE_WIDGET_KEY,
    CachedUploadedFile,
    anonymize_enabled,
    clear_uploaded_files,
    ingest_widget_files,
    registry_entries,
    remove_file,
    set_anonymize_enabled,
    uploaded_files,
    uploader_key,
)

# Injected once per page. Streamlit's sidebar can outgrow the viewport (Options panel plus
# a dozen section checkboxes plus a quiz selector) and strand the lower controls below the
# fold with no way to scroll to them. Target every testid Streamlit has used for the
# sidebar's scroll container across versions so this doesn't silently stop working on an
# upgrade, and separately bound the selectbox popover and the uploaded-files list, both of
# which grow with the number of uploaded quizzes.
_SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"] {
    overflow-y: auto !important;
}
section[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    overflow-y: auto !important;
    max-height: 100vh !important;
}
/* Selectbox dropdowns are portaled to <body> as fixed-position elements, so no ancestor
   bounds them: with enough uploaded files the option list is tall enough that BaseWeb
   places its lower rows past the bottom of the window, where they can't be scrolled to.
   Capping the inner scroller in viewport units keeps the whole popover on screen and
   scrollable within that cap. (The list is a `div` with role="listbox", not a `ul`.) */
div[data-testid="stSelectboxVirtualDropdown"] > div,
div[data-baseweb="popover"] div[role="listbox"] {
    max-height: 40vh !important;
    overflow-y: auto !important;
}
/* The uploaded-files list inside its expander — one row per file, so a large upload batch
   would otherwise stretch the sidebar arbitrarily far. */
[data-testid="stSidebarUserContent"] [data-testid="stExpanderDetails"] {
    max-height: 30vh;
    overflow-y: auto;
}
</style>
"""


def inject_sidebar_css() -> None:
    st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)


def render_options_panel() -> tuple[list[CachedUploadedFile], bool]:
    """The shared sidebar "Options" panel, identical on every page.

    Uploads land in the shared registry (`analytics.upload_cache`) rather than being read
    off the widget, so all pages see the same quizzes with no re-upload and no per-page
    copies. Returns (uploaded files, anonymize flag).

    Widget keys here are fixed rather than per-page: only one page runs per script run, so
    there is no collision, and sharing the key is what makes the anonymize choice persist
    across a page switch.
    """
    st.sidebar.title("Options")
    widget_files = st.sidebar.file_uploader(
        "Upload responses file(s)",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
        help="Upload one or more Moodle responses exports in CSV, XLS, or XLSX format. Files uploaded on any page are available on all of them.",
        key=uploader_key(),
    )
    ingest_widget_files(widget_files)

    entries = registry_entries()
    if entries:
        with st.sidebar.expander(f"📎 Uploaded Files ({len(entries)})", expanded=False):
            for file_id, display_name in entries:
                name_col, remove_col = st.columns([6, 1], vertical_alignment="center")
                name_col.write(display_name)
                if remove_col.button(
                    "✕",
                    key=f"remove_upload_{file_id}",
                    help=f"Remove {display_name}",
                ):
                    remove_file(file_id)
                    # Derived data (parsed frames, metrics, TED matrices) is cached by
                    # content, so a removal changes the cache key rather than leaving a
                    # stale hit — but clearing keeps memory from growing across edits.
                    st.cache_data.clear()
                    st.rerun()

    if st.sidebar.button(
        "🗑️ Clear / Reset All Uploaded Files",
        use_container_width=True,
        key="clear_all_uploads",
    ):
        clear_uploaded_files()
        st.cache_data.clear()
        st.rerun()

    # Reads its default from the plain session key and writes back to it, so the choice
    # survives a page switch (see the note on ANONYMIZE_STATE_KEY).
    anonymize_data = st.sidebar.checkbox(
        "🔒 Anonymize Student Data",
        value=anonymize_enabled(),
        key=ANONYMIZE_WIDGET_KEY,
    )
    set_anonymize_enabled(anonymize_data)
    return uploaded_files(), anonymize_data


def load_shared_response_df(
    files: list[CachedUploadedFile],
    anonymize_data: bool,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[str]]:
    """Parse the shared registry into (quiz metadata, response frame, quiz names).

    The registry stores name + bytes rather than a parsed frame per file on purpose:
    `load_quiz_data` parses the whole set together — that is where a Responses export and
    a Grades-with-breakdown export for the *same* quiz get merged into one frame — so a
    per-file parse would not compose to the same result. The parse itself is cached by file
    content, so this stays a cache hit across pages and reruns.
    """
    if not files:
        return [], pd.DataFrame(), []

    quiz_metadata, response_df = load_quiz_data(files)
    if anonymize_data:
        response_df = anonymize_response_df(response_df)
    quiz_names = [item["quiz_name"] for item in quiz_metadata] if not response_df.empty else []
    return quiz_metadata, response_df, quiz_names
