from __future__ import annotations

import re

import pandas as pd


# PRT names are author-defined in STACK (default "prt1"/"prt2", but Moodle exports
# commonly show custom names like "Result"/"Result2" instead), so a segment is
# recognized as a PRT field by process of elimination rather than a literal "prt"
# prefix: exclude the "Seed: ..." metadata field and "ansK: ... [tag]" fields, and
# treat anything else shaped like "<name>: value" as a PRT.
_ANS_FIELD_RE = re.compile(r"^\s*ans\d+\s*:\s*.*\[(?:score|valid|invalid)\]\s*$", re.IGNORECASE)
_SEED_FIELD_RE = re.compile(r"^\s*seed\s*:", re.IGNORECASE)


def _parse_prt_values(response_text: str) -> list[tuple[str, float, str]]:
    """Extract PRT values from a response string."""
    if not response_text:
        return []

    prts: list[tuple[str, float, str]] = []
    for part in response_text.split(";"):
        if _ANS_FIELD_RE.match(part) or _SEED_FIELD_RE.match(part):
            continue
        match = re.match(r"^\s*(\w+)\s*:\s*(.+)$", part)
        if not match:
            continue
        prt_name = match.group(1).lower()
        value = match.group(2).strip()
        if value == "!":
            prts.append((prt_name, 0.0, "syntax_error"))
            continue

        score_match = re.search(r"#\s*=\s*([\d.]+)", value)
        if score_match:
            score = float(score_match.group(1))
            status = "correct" if score >= 0.5 else "incorrect"
            prts.append((prt_name, score, status))
            continue

        lower_value = value.lower()
        if any(token in lower_value for token in ["correct", "true", "pass"]):
            prts.append((prt_name, 1.0, "correct"))
        elif any(token in lower_value for token in ["incorrect", "false", "fail"]):
            prts.append((prt_name, 0.0, "incorrect"))
        else:
            prts.append((prt_name, 0.0, "incorrect"))

    return prts


def compute_prt_pass_rates(response_df: pd.DataFrame) -> pd.DataFrame:
    """Compute PRT pass rates by question and PRT name."""
    if response_df.empty:
        return pd.DataFrame(columns=["question", "prt_name", "attempts", "pass_rate"])

    rows = []
    for question, group in response_df.groupby("question"):
        for prt_name in sorted(group["prt_name"].dropna().astype(str).unique()):
            prt_rows = group[group["prt_name"] == prt_name]
            attempts = len(prt_rows)
            pass_rate = 0.0
            if attempts:
                pass_rate = round(float((prt_rows["prt_score"] >= 0.5).mean()) * 100, 2)
            rows.append({"question": question, "prt_name": prt_name, "attempts": attempts, "pass_rate": pass_rate})

    return pd.DataFrame(rows)


def build_prt_frame(response_df: pd.DataFrame) -> pd.DataFrame:
    """Create a per-question, per-PRT frame for downstream charts."""
    if response_df.empty:
        return pd.DataFrame(columns=["question", "prt_name", "prt_score", "response_status"])

    if {"prt_name", "prt_score"}.issubset(response_df.columns):
        return response_df[["question", "prt_name", "prt_score", "response_status"]].copy()

    rows: list[dict[str, object]] = []
    for _, row in response_df.iterrows():
        parsed = _parse_prt_values(str(row.get("response_text", "")))
        if not parsed:
            rows.append({"question": row["question"], "prt_name": "prt1", "prt_score": 0.0, "response_status": row.get("response_status", "incorrect")})
            continue
        for prt_name, prt_score, status in parsed:
            rows.append({"question": row["question"], "prt_name": prt_name, "prt_score": prt_score, "response_status": status})

    return pd.DataFrame(rows)
