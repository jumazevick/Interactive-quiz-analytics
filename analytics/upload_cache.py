from __future__ import annotations

from typing import Any

import streamlit as st

SHARED_UPLOAD_KEY = "shared_uploaded_files"
UPLOADER_KEY_COUNTER = "uploader_key_counter"


class CachedUploadedFile:
    """Lightweight file-like stand-in for Streamlit's UploadedFile, reconstructed from
    bytes cached in session_state so an upload survives a page switch.

    Deliberately does NOT subclass/wrap io.BytesIO with a `.name` attribute: Streamlit's
    cache hasher treats any file-like object that has a `.name` as a real on-disk file
    and calls os.path.getmtime(obj.name) on it, which raises FileNotFoundError for a
    synthetic name. A plain class sidesteps that special case entirely.
    """

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self._pos = 0

    def getvalue(self) -> bytes:
        return self._data

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk = self._data[self._pos:]
            self._pos = len(self._data)
        else:
            chunk = self._data[self._pos:self._pos + size]
            self._pos += len(chunk)
        return chunk

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = len(self._data) + pos
        return self._pos

    def tell(self) -> int:
        return self._pos


# Pass into any @st.cache_data(hash_funcs=CACHE_HASH_FUNCS) whose arguments might
# include CachedUploadedFile instances — Streamlit has no built-in way to hash a
# custom class, so without this a cached function call would raise UnhashableParamError.
CACHE_HASH_FUNCS = {CachedUploadedFile: lambda f: (f.name, f.getvalue())}


def sync_uploaded_files(freshly_uploaded: list[Any] | None) -> tuple[list[Any], bool]:
    """Persist uploaded files in session_state so they survive a page switch.

    Streamlit resets each page's file_uploader widget when navigating to a
    different page file, so without this, switching between the Question
    Analysis and Quiz Analysis sections would silently drop whatever was just
    uploaded. If the user just uploaded something new on this page, cache it
    for the other page; otherwise fall back to the most recently cached files
    (from either page).

    Returns (resolved_files, used_cache).
    """
    if freshly_uploaded:
        st.session_state[SHARED_UPLOAD_KEY] = [
            {"name": f.name, "data": f.getvalue()} for f in freshly_uploaded
        ]
        return list(freshly_uploaded), False

    cached = st.session_state.get(SHARED_UPLOAD_KEY)
    if cached:
        restored = [CachedUploadedFile(item["name"], item["data"]) for item in cached]
        return restored, True

    return freshly_uploaded or [], False


def get_uploader_key() -> str:
    """Widget key for the shared file_uploader. Streamlit's file_uploader can't be
    cleared by writing to its session_state value directly — the only reliable way is
    to swap in a new `key`, which makes it a "new" widget with empty state."""
    counter = st.session_state.get(UPLOADER_KEY_COUNTER, 0)
    return f"shared_file_uploader_{counter}"


def clear_uploaded_files() -> None:
    """Reset the shared upload cache and force the file_uploader widget to reinitialize
    empty on the next rerun, fixing the case where navigating away and back leaves a
    visually-empty uploader silently backed by cached files with no way to remove them."""
    st.session_state[UPLOADER_KEY_COUNTER] = st.session_state.get(UPLOADER_KEY_COUNTER, 0) + 1
    st.session_state.pop(SHARED_UPLOAD_KEY, None)
