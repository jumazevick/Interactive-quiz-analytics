from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st

# The single source of truth for uploaded quizzes, shared by every page: an
# insertion-ordered dict of {file_id: {"name": str, "data": bytes}}. Nothing downstream
# reads st.file_uploader's return value directly — the widget is an ingestion point that
# feeds this registry, and the registry is what every page renders and parses from.
REGISTRY_KEY = "uploaded_quizzes"

# Ids of files the user removed. Belt-and-braces against Streamlit's file_uploader, which
# keeps returning its full list on every rerun and would otherwise resurrect a removed
# file. In practice the nonce bump below already prevents that (the widget is recreated
# empty), so this only matters if the widget somehow reports files again anyway.
TOMBSTONE_KEY = "removed_upload_ids"

# Bumped on any removal or reset. st.file_uploader can't be cleared by writing to its
# session_state value — swapping in a new `key` is the only reliable way to make it
# reinitialize empty.
NONCE_KEY = "uploader_nonce"

# The anonymize choice is global state, so it lives under a plain session_state key and
# the checkbox is only a view onto it. A widget key alone is not enough: Streamlit
# garbage-collects widget state for widgets that weren't rendered on the previous script
# run, and a page switch counts — so a checkbox keyed only by ANONYMIZE_WIDGET_KEY silently
# reverted to its default every time the user changed page (verified in the browser).
# Plain (non-widget) keys are never collected, so this one carries the value across.
ANONYMIZE_STATE_KEY = "anonymize_student_data"
ANONYMIZE_WIDGET_KEY = "anonymize_student_data_toggle"


def anonymize_enabled() -> bool:
    return bool(st.session_state.get(ANONYMIZE_STATE_KEY, True))


def set_anonymize_enabled(enabled: bool) -> None:
    st.session_state[ANONYMIZE_STATE_KEY] = bool(enabled)


class CachedUploadedFile:
    """Lightweight file-like stand-in for Streamlit's UploadedFile, reconstructed from
    the bytes held in the registry so an upload survives a page switch.

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


def _registry() -> dict[str, dict[str, Any]]:
    return st.session_state.setdefault(REGISTRY_KEY, {})


def _tombstones() -> set[str]:
    return st.session_state.setdefault(TOMBSTONE_KEY, set())


def _file_id(name: str, data: bytes) -> str:
    """Stable id for one uploaded file: its name, or name + a short content hash when a
    *different* file arrives under a name that is already taken (so two same-named files
    from different folders both survive instead of one silently replacing the other).
    Re-uploading the identical bytes under the same name keeps the original id."""
    registry = _registry()
    existing = registry.get(name)
    if existing is None or existing["data"] == data:
        return name
    return f"{name}#{hashlib.sha1(data).hexdigest()[:8]}"


def uploader_key() -> str:
    """Widget key for the shared file_uploader, carrying the current nonce."""
    return f"shared_file_uploader_{st.session_state.get(NONCE_KEY, 0)}"


def ingest_widget_files(widget_files: list[Any] | None) -> None:
    """Merge anything new from the uploader widget into the registry.

    An empty widget means nothing can be resurrected on this run, which is also the point
    at which pending tombstones can safely be dropped — so re-uploading a file the user
    removed earlier works normally.
    """
    if not widget_files:
        st.session_state[TOMBSTONE_KEY] = set()
        return

    registry = _registry()
    tombstones = _tombstones()
    for uploaded in widget_files:
        data = uploaded.getvalue()
        file_id = _file_id(uploaded.name, data)
        if file_id in tombstones or file_id in registry:
            continue
        registry[file_id] = {"name": uploaded.name, "data": data}


def registry_entries() -> list[tuple[str, str]]:
    """(file_id, display name) for every uploaded quiz, in upload order."""
    return [(file_id, entry["name"]) for file_id, entry in _registry().items()]


def uploaded_files() -> list[CachedUploadedFile]:
    """Every uploaded quiz as a file-like object, in upload order. This — never the
    widget's return value — is what the pages parse."""
    return [CachedUploadedFile(entry["name"], entry["data"]) for entry in _registry().values()]


def remove_file(file_id: str) -> None:
    """Drop one file from the registry so it disappears from every page at once."""
    _registry().pop(file_id, None)
    _tombstones().add(file_id)
    st.session_state[NONCE_KEY] = st.session_state.get(NONCE_KEY, 0) + 1


def clear_uploaded_files() -> None:
    """Full reset: registry, tombstones, and a fresh uploader widget."""
    st.session_state[REGISTRY_KEY] = {}
    st.session_state[TOMBSTONE_KEY] = set()
    st.session_state[NONCE_KEY] = st.session_state.get(NONCE_KEY, 0) + 1
