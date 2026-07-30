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

# Widget key for the Uploaded Files expander — shared between the panel's own use and
# the explicit re-assertion after a removal (see render_options_panel).
_UPLOADED_FILES_EXPANDER_KEY = "uploaded_files_expander_open"

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
/* Keep the sidebar's own height in check, because a selectbox near the bottom of a long
   sidebar is unusable: its option list is portaled out as a fixed-position element and
   positioned *below* the trigger, so with the trigger at y≈980 in a 1050px window the
   whole list opens off-screen and can't be scrolled to — the popover is outside the
   sidebar's scroll container, so scrolling the sidebar afterwards doesn't bring it back.
   (Reproduced with 14 uploaded quizzes before these caps were added.) Two things drive
   that height: the uploader's own chip per uploaded file, and our uploaded-files list. */
section[data-testid="stSidebar"] [data-testid="stFileChips"] {
    max-height: 9vh;
    overflow-y: auto;
}
/* The uploaded-files list: tall enough to show a couple of full entries rather than
   clipping one mid-name, with the names a step down in size since a Moodle export
   filename is long and wraps over several lines at body size. */
[data-testid="stSidebarUserContent"] [data-testid="stExpanderDetails"] {
    max-height: 24vh;
    overflow-y: auto;
}
[data-testid="stSidebarUserContent"] [data-testid="stExpanderDetails"] p {
    font-size: 0.78rem;
    line-height: 1.35;
    word-break: break-word;
}
/* The per-file remove button: square, and centred on the row rather than sitting against
   the top of a name that wrapped onto three lines. */
[data-testid="stSidebarUserContent"] [data-testid="stExpanderDetails"] [data-testid="stButton"] {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}
[data-testid="stSidebarUserContent"] [data-testid="stExpanderDetails"] [data-testid="stButton"] button {
    min-width: 2rem;
    padding: 0.15rem 0.4rem;
}
/* Multiselect chips in the sidebar (quizzes to include, statistics, metrics): the same
   long filenames, so shrink them too — otherwise five selected quizzes fill the sidebar
   with truncated look-alike chips. */
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] span[data-baseweb="tag"],
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] [role="option"],
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] div[role="button"] {
    font-size: 0.74rem;
}
/* The chip area caps itself at ~155px and hides the overflow, which clips the fifth
   selected quiz in half. Raise the cap — the inner wrapper already scrolls, so a longer
   list stays reachable rather than being cut off. */
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    max-height: 15rem !important;
}
/* Scroll headroom under the last control, for two reasons. It stops the final checkbox
   sitting flush against the bottom edge, and react-aria only flips a selectbox's option
   list above its trigger when it judges there to be too little room below — at a tall
   viewport it happily opens a scrollable list that runs past the bottom edge instead. The
   padding lets the sidebar scroll any control up into the top half of the window, where
   the list always has room. */
[data-testid="stSidebarUserContent"] {
    padding-bottom: 45vh;
}
/* And cap the option list itself so a low trigger still leaves it mostly on screen.
   Selectors cover both renderers Streamlit has shipped — react-aria (current, plain
   role="listbox") and the older BaseWeb popover — since neither is a documented API and
   the markup has changed across versions. */
div[role="listbox"],
div[data-testid="stSelectboxVirtualDropdown"] > div,
div[data-baseweb="popover"] div[role="listbox"] {
    max-height: 20vh !important;
    overflow-y: auto !important;
}
</style>
"""


def inject_sidebar_css() -> None:
    st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)


def render_sidebar_bottom_spacer() -> None:
    """A block of real, rendered height at the very end of the sidebar, so the last
    widget isn't flush against the bottom edge of the window.

    `padding-bottom` on the sidebar's own scroll container (see `_SIDEBAR_CSS` above)
    does not reliably create visible space here: Streamlit wraps the sidebar's actual
    scrolling element (`stSidebarContent`) around a child (`stSidebarUserContent`) whose
    own height is separately constrained, so padding added to that inner child doesn't
    count towards the outer element's scrollable area (confirmed by scrolling a page to
    its true end with padding applied — the last checkbox still sits flush against the
    edge). An actual rendered element, on the other hand, always counts, since it's real
    layout content rather than a padding region on a possibly-clipped box.

    Call this once, as the very last thing added to the sidebar on each page — after any
    page-specific controls that follow the shared Options panel — not from inside
    `render_options_panel()` itself, since that panel usually isn't the last thing in the
    sidebar.
    """
    st.sidebar.markdown('<div style="height: 2.5rem;"></div>', unsafe_allow_html=True)


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
        # A stable `key` makes Streamlit track this expander's open/closed state in
        # session_state on its own, the same way it does for a checkbox — `expanded=`
        # below is then only its default the very first time the key is seen, exactly
        # the "Select All"/"Deselect All" convention already used for the section
        # checkboxes elsewhere on these pages. Without a key, an `st.expander` has no
        # persistent state at all: it silently reverts to `expanded=` on every rerun,
        # which is what collapsed this panel out from under the user on every removal —
        # each click on a file's own ✕ triggers `st.rerun()` below.
        with st.sidebar.expander(
            f"📎 Uploaded Files ({len(entries)})",
            expanded=False,
            key=_UPLOADED_FILES_EXPANDER_KEY,
        ):
            for file_id, display_name in entries:
                name_col, remove_col = st.columns([5, 1], vertical_alignment="center")
                name_col.write(display_name)
                if remove_col.button(
                    "✕",
                    key=f"remove_upload_{file_id}",
                    help=f"Remove {display_name}",
                ):
                    remove_file(file_id)
                    # The click that just fired came from inside the expander, so the
                    # user plainly had it open — re-assert that explicitly rather than
                    # trust the key-tracked state alone, so a batch of several removals
                    # in a row can't have the panel snap shut between clicks for any
                    # reason (e.g. a future Streamlit version changing how a widget's
                    # tracked state behaves when its container's content changes shape
                    # around it, since here the row that just disappeared came from
                    # inside the very panel we're trying to keep open).
                    st.session_state[_UPLOADED_FILES_EXPANDER_KEY] = True
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
