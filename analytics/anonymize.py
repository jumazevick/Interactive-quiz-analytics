from __future__ import annotations

import pandas as pd


def anonymize_response_df(response_df: pd.DataFrame) -> pd.DataFrame:
    """Replace real student names/emails with stable per-student pseudonyms.

    Applied once, immediately after loading, so every downstream table, chart, and
    PDF export (which all derive from this same response_df) is anonymized for free —
    no need to touch each individual display point separately. student_id keeps acting
    as a valid grouping/join key since the mapping is a consistent 1:1 relabeling, not
    a shuffle.
    """
    if response_df.empty or "student_id" not in response_df.columns:
        return response_df

    anonymized = response_df.copy()
    original_ids = anonymized["student_id"].astype(str)
    unique_ids = sorted(original_ids.unique())
    name_map = {orig: f"Student {i + 1}" for i, orig in enumerate(unique_ids)}
    email_map = {orig: f"student{i + 1}@anonymized.edu" for i, orig in enumerate(unique_ids)}

    if "student_name" in anonymized.columns:
        anonymized["student_name"] = original_ids.map(name_map)
    anonymized["student_id"] = original_ids.map(email_map)
    return anonymized
