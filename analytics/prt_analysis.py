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
        frame = response_df[["question", "prt_name", "prt_score", "response_status"]].copy()
        frame["has_prt"] = True
        return frame

    rows: list[dict[str, object]] = []
    for _, row in response_df.iterrows():
        parsed = _parse_prt_values(str(row.get("response_text", "")))
        if not parsed:
            # A response with no PRT trace still contributes a scored-zero row, because the
            # pass rates are per *attempt* and a blank/invalid response is a failed attempt.
            # `has_prt` marks it as synthesized so the heatmap can tell a question that has
            # no Potential Response Tree at all from one whose PRT everybody failed.
            rows.append({"question": row["question"], "prt_name": "prt1", "prt_score": 0.0, "response_status": row.get("response_status", "incorrect"), "has_prt": False})
            continue
        for prt_name, prt_score, status in parsed:
            rows.append({"question": row["question"], "prt_name": prt_name, "prt_score": prt_score, "response_status": status, "has_prt": True})

    return pd.DataFrame(rows)


# Cells for a question that has no PRT at all. Plotly renders NaN cells as transparent, so
# painting the plot area this colour is what makes them read as "not applicable" rather
# than as a zero pass rate — which the red end of the pass/fail scale would otherwise
# claim, making an un-PRT'd question look like a total failure.
NO_PRT_CELL_COLOR = "#d4d4d8"


def build_prt_pass_heatmap(
    prt_pass_rates: pd.DataFrame,
    question_order: list,
    prt_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Question x PRT pass-rate matrix for the heatmap.

    Every question in `question_order` gets a row even if it has no PRT data, and missing
    cells stay NaN rather than being filled with 0 — see `NO_PRT_CELL_COLOR`.

    Pass `prt_frame` to blank out questions with no Potential Response Tree at all. Those
    otherwise show a 0% pass rate, because `build_prt_frame` scores their responses through
    a synthesized `prt1` — correct for a per-attempt pass rate, but it paints an un-PRT'd
    question the same red as one every student failed. Blanking happens here, at display
    time; the pass rates themselves are untouched.
    """
    if prt_pass_rates.empty:
        return pd.DataFrame(index=list(question_order))

    heatmap_df = prt_pass_rates.pivot_table(
        index="question",
        columns="prt_name",
        values="pass_rate",
        aggfunc="first",
        dropna=False,
    ).reindex(list(question_order))

    if prt_frame is not None and "has_prt" in prt_frame.columns:
        with_prt = set(prt_frame.loc[prt_frame["has_prt"].astype(bool), "question"])
        without_prt = [q for q in heatmap_df.index if q not in with_prt]
        if without_prt:
            heatmap_df.loc[without_prt] = float("nan")

    return heatmap_df
