from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd


def _fix_mojibake(text: str) -> str:
    """Undo UTF-8-decoded-as-Latin-1 mangling (e.g. a middle dot '·' surviving a round trip
    through the wrong codec as 'Â·', or 'π' splitting into 'Ï' plus a stray control byte) that
    some Moodle export pipelines bake into the CSV/Excel file before we ever see it.

    Safe no-op on text that's already correct: re-encoding to Latin-1 raises for any character
    above U+00FF (i.e. a properly decoded non-ASCII symbol), and decoding those Latin-1 bytes as
    UTF-8 raises unless they happen to form a genuine mojibake byte sequence — so plain ASCII and
    correctly-decoded Unicode both pass through unchanged."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _fix_mojibake_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: _fix_mojibake(v) if isinstance(v, str) else v)
    return df


def _clean_field(value: Any) -> str:
    """Blank Moodle export cells parse to NaN, which is truthy in Python — an
    `x or default` chain never falls through for them. Normalize to "" so callers can
    use plain `or` fallbacks safely."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _resolve_student_identity(row: "pd.Series", index: int) -> tuple[str, str]:
    """Derive (student_id, student_name) for a response row, falling back to a stable,
    unique per-row placeholder when Moodle's export has stripped name/email fields
    entirely (e.g. a site-level "anonymize" download option) — otherwise every blank
    row would collapse onto the same student_id / "Anonymized Student" name, merging
    distinct students together in every downstream table and chart."""
    email = _clean_field(row.get("Email address")) or _clean_field(row.get("anonymized_full_name"))
    first_name = _clean_field(row.get("First name"))
    surname = _clean_field(row.get("Surname")) or _clean_field(row.get("Last name"))
    full_name = f"{first_name} {surname}".strip()

    student_id = email or f"student{index + 1}@anonymized.local"
    student_name = full_name or f"Student {index + 1}"
    return str(student_id), student_name


# Canonical Moodle export header -> alternate names seen from other export
# configurations/languages/plugins (e.g. a "Username" column instead of "Email
# address", "Status" instead of "State"). Keys are matched case-insensitively
# and only used to fill in a canonical column that isn't already present, so a
# real "Email address" column always wins over a "Username" alias.
_COLUMN_ALIASES: dict[str, list[str]] = {
    "Email address": ["email address", "email", "e-mail", "e-mail address", "username", "user name"],
    "First name": ["first name", "given name", "firstname"],
    "Surname": ["surname", "last name", "lastname", "family name"],
    "State": ["state", "status"],
    "Started on": ["started on", "started", "start time", "time started"],
    "Completed": ["completed", "completion time", "time completed", "end time"],
    "Time taken": ["time taken", "duration"],
}


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to their canonical name whenever a recognized alias is used
    instead (e.g. a Moodle site/plugin exporting "Username"/"Status" rather than
    "Email address"/"State"), so every downstream function can keep relying on one
    fixed set of header names regardless of which export produced the file."""
    existing_lower = {str(col).strip().lower() for col in df.columns}
    rename_map: dict[str, str] = {}

    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical.lower() in existing_lower:
            continue
        for col in df.columns:
            if col in rename_map:
                continue
            if str(col).strip().lower() in aliases:
                rename_map[col] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _clean_html_text(text: Any) -> str:
    """Strip HTML tags/entities from a Moodle question/answer cell, leaving LaTeX delimiters intact."""
    if pd.isna(text):
        return ""
    cleaned = html.unescape(str(text))
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_uploaded_file(file_obj: Any) -> pd.DataFrame:
    """Load a Moodle export into a normalized DataFrame."""
    if file_obj.name.endswith(".xls"):
        df = pd.read_excel(file_obj, engine="xlrd")
    elif file_obj.name.endswith(".xlsx"):
        df = pd.read_excel(file_obj, engine="openpyxl")
    elif file_obj.name.endswith(".csv"):
        df = pd.read_csv(file_obj, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file format: {file_obj.name}")

    df = _fix_mojibake_df(df)
    df = _normalize_column_names(df)

    return df


def normalize_question_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clone the input with a normalized set of question columns for analytics."""
    normalized_df = df.copy()
    question_columns = [col for col in normalized_df.columns if re.match(r"^(?:Q|Question|q)\.?\s*\d+", col) or re.match(r"^Response\s*\d+", col)]

    if not question_columns:
        return normalized_df

    for column in question_columns:
        if column.startswith("Q") or column.startswith("q"):
            normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")

    return normalized_df


def detect_export_type(df: pd.DataFrame) -> str:
    """Detect whether a Moodle export is a Responses export or a Grades-with-breakdown export."""
    normalized_columns = {str(col).strip().lower() for col in df.columns}
    has_response_columns = any(re.match(r"^response\s*\d+$", col, re.IGNORECASE) for col in df.columns)
    has_breakdown_columns = any(re.match(r"^(?:q|question)\.?(?:\s*)\d+\s*/", col, re.IGNORECASE) for col in df.columns)
    has_grade_column = any(re.match(r"^grade/\d+", col, re.IGNORECASE) for col in df.columns)

    if has_breakdown_columns and has_grade_column and not has_response_columns:
        return "grades_breakdown"
    if has_response_columns:
        return "responses"
    return "unknown"


def parse_response_cell(cell_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a single Response cell to extract ansK and prtK values."""
    if pd.isna(cell_text) or not isinstance(cell_text, str):
        return [], []

    # 1. Parse ans fields: ansK: expression [tag]
    ans_matches = re.finditer(r"ans(\d+):\s*(.*?)\s*\[(score|valid|invalid)\]", cell_text)
    ans_list = []
    for m in ans_matches:
        ans_list.append({
            "index": int(m.group(1)),
            "expression": m.group(2).strip(),
            "tag": m.group(3)
        })

    # 2. Parse prt fields: prtK: ! OR prtK: # = fraction | note1 | note2...
    prt_matches = re.finditer(r"prt(\d+):\s*(!|# = ([0-9.]+))(?:\s*\|\s*([^;]*))?", cell_text)
    prt_list = []
    for m in prt_matches:
        idx = int(m.group(1))
        val = m.group(2)
        fraction = None
        answer_note = "(invalid/blank input)"

        if val != "!":
            fraction = float(m.group(3))
            notes_str = m.group(4)
            if notes_str:
                tokens = [t.strip() for t in notes_str.split("|") if t.strip()]
                if tokens:
                    answer_note = tokens[-1]
                else:
                    answer_note = ""
            else:
                answer_note = ""

        prt_list.append({
            "index": idx,
            "fraction": fraction,
            "answer_note": answer_note
        })

    return ans_list, prt_list


def parse_completed_dt(val: Any) -> pd.Timestamp:
    """Parse Completed date string into Timestamp for tie-breaking."""
    try:
        dt = pd.to_datetime(val)
        if pd.isna(dt):
            return pd.Timestamp.min
        return dt
    except Exception:
        return pd.Timestamp.min


def build_response_rows(df: pd.DataFrame, quiz_name: str) -> pd.DataFrame:
    """Convert a Moodle Responses export into a flattened question-response table."""
    # Row filtering: Drop any row where State is not exactly "Finished"
    state_cols = [col for col in df.columns if str(col).strip().lower() == "state"]
    if state_cols:
        df = df[df[state_cols[0]].astype(str).str.strip() == "Finished"]

    if df.empty:
        return pd.DataFrame(columns=[
            "student_id", "student_name", "question", "grade", "max_grade",
            "response_status", "response_text", "quiz_name", "ans_list",
            "prt_list", "overall_grade", "completed_dt", "started_on",
            "attempt_idx", "source_type", "question_text", "right_answer_text"
        ])

    # Identify response columns (Response 1 ... Response N)
    response_cols = [col for col in df.columns if re.match(r"^Response\s*\d+", col, re.IGNORECASE)]
    if not response_cols:
        # Fallback to general question columns if not response-based
        response_cols = [col for col in df.columns if re.match(r"^(?:Q|Question|q)\.?\s*\d+", col, re.IGNORECASE)]

    def get_col_number(col):
        m = re.search(r"\d+", col)
        return int(m.group(0)) if m else 0

    response_cols = sorted(response_cols, key=get_col_number)

    # Optional display-metadata columns from Moodle's Responses report Display options
    # ("Question text" / "Right answer"). Purely additional context for the UI — never
    # used for scoring, which stays driven entirely by the ans/prt tags in Response i.
    question_text_cols = {
        get_col_number(col): col
        for col in df.columns
        if re.match(r"^Question\s*\d+$", str(col).strip(), re.IGNORECASE)
    }
    right_answer_cols = {
        get_col_number(col): col
        for col in df.columns
        if re.match(r"^Right answer\s*\d+$", str(col).strip(), re.IGNORECASE)
    }

    # Determine M (number of PRT parts) for each question column
    M_dict = {}
    for col in response_cols:
        max_k = 1
        for cell in df[col].dropna():
            _, prt_list = parse_response_cell(str(cell))
            for prt in prt_list:
                if prt["index"] > max_k:
                    max_k = prt["index"]
        M_dict[col] = max_k

    # Identify the Grade column (e.g. Grade/10.00)
    grade_col = None
    for col in df.columns:
        if re.match(r"^Grade/\d+", col, re.IGNORECASE):
            grade_col = col
            break

    records = []
    for index, row in df.iterrows():
        student_id, student_name = _resolve_student_identity(row, index)
        completed_raw = row.get("Completed", "")
        started_raw = row.get("Started on") or row.get("Started") or completed_raw
        completed_dt = parse_completed_dt(completed_raw)
        started_dt = parse_completed_dt(started_raw)

        overall_grade = 0.0
        if grade_col:
            val = pd.to_numeric(row.get(grade_col), errors="coerce")
            if pd.notna(val):
                overall_grade = float(val)

        for col in response_cols:
            q_num = get_col_number(col)
            question_label = f"Q{q_num}"
            cell_text = str(row[col]) if pd.notna(row[col]) else ""

            question_text = _clean_html_text(row.get(question_text_cols.get(q_num))) if q_num in question_text_cols else ""
            right_answer_text = _clean_html_text(row.get(right_answer_cols.get(q_num))) if q_num in right_answer_cols else ""

            ans_list, prt_list = parse_response_cell(cell_text)

            is_blank = len(ans_list) == 0
            is_invalid = any(ans["tag"] == "invalid" for ans in ans_list)

            # Score computation: mean over all PRTs K of (prtK.fraction or 0.0)
            M = M_dict[col]
            prt_map = {prt["index"]: prt for prt in prt_list}
            prt_fractions = []
            for k in range(1, M + 1):
                if k in prt_map and prt_map[k]["fraction"] is not None:
                    prt_fractions.append(prt_map[k]["fraction"])
                else:
                    prt_fractions.append(0.0)

            q_score = sum(prt_fractions) / M if M > 0 else 0.0

            # Classification
            if is_blank:
                response_status = "blank"
            elif is_invalid:
                response_status = "invalid"
            elif q_score == 1.0:
                response_status = "correct"
            else:
                response_status = "incorrect"

            records.append({
                "student_id": str(student_id),
                "student_name": student_name,
                "question": question_label,
                "grade": q_score,
                "max_grade": 1.0,
                "response_status": response_status,
                "response_text": cell_text,
                "quiz_name": quiz_name,
                "ans_list": ans_list,
                "prt_list": prt_list,
                "overall_grade": overall_grade,
                "completed_dt": completed_dt,
                "started_on": started_dt,
                "attempt_idx": index,
                "source_type": "responses",
                "question_text": question_text,
                "right_answer_text": right_answer_text,
            })

    return pd.DataFrame(records)


def build_grade_breakdown_rows(df: pd.DataFrame, quiz_name: str) -> pd.DataFrame:
    """Convert a Moodle Grades-with-breakdown export into a flattened question-score table."""
    state_cols = [col for col in df.columns if str(col).strip().lower() == "state"]
    if state_cols:
        df = df[df[state_cols[0]].astype(str).str.strip() == "Finished"]

    if df.empty:
        return pd.DataFrame(columns=[
            "student_id", "student_name", "question", "grade", "max_grade",
            "response_status", "response_text", "quiz_name", "overall_grade",
            "completed_dt", "started_on", "attempt_idx", "source_type",
            "raw_score", "question_max_score", "question_text", "right_answer_text"
        ])

    question_cols = [
        col for col in df.columns
        if re.match(r"^(?:Q|Question|q)\.?(?:\s*)\d+\s*/", str(col), re.IGNORECASE)
    ]

    def get_col_number(col: str) -> int:
        m = re.search(r"\d+", str(col))
        return int(m.group(0)) if m else 0

    question_cols = sorted(question_cols, key=get_col_number)

    grade_col = None
    for col in df.columns:
        if re.match(r"^Grade/\d+", str(col), re.IGNORECASE):
            grade_col = col
            break

    records = []
    for index, row in df.iterrows():
        student_id, student_name = _resolve_student_identity(row, index)
        completed_raw = row.get("Completed", "")
        started_raw = row.get("Started on") or row.get("Started") or completed_raw
        completed_dt = parse_completed_dt(completed_raw)
        started_dt = parse_completed_dt(started_raw)

        overall_grade = 0.0
        if grade_col:
            val = pd.to_numeric(row.get(grade_col), errors="coerce")
            if pd.notna(val):
                overall_grade = float(val)

        for col in question_cols:
            q_num = get_col_number(col)
            question_label = f"Q{q_num}"
            raw_value = row.get(col)
            raw_score = pd.to_numeric(raw_value, errors="coerce")
            max_score_match = re.search(r"/(\d+(?:\.\d+)?)", str(col))
            max_score = float(max_score_match.group(1)) if max_score_match else 0.0
            if pd.isna(raw_score) or max_score <= 0:
                grade = 0.0
                response_status = "blank"
            else:
                grade = float(raw_score / max_score) if max_score > 0 else 0.0
                response_status = "correct" if grade >= 1.0 else "incorrect"

            records.append({
                "student_id": str(student_id),
                "student_name": student_name,
                "question": question_label,
                "grade": grade,
                "max_grade": 1.0,
                "response_status": response_status,
                "response_text": "",
                "quiz_name": quiz_name,
                "overall_grade": overall_grade,
                "completed_dt": completed_dt,
                "started_on": started_dt,
                "attempt_idx": index,
                "source_type": "grades_breakdown",
                "raw_score": float(raw_score) if pd.notna(raw_score) else None,
                "question_max_score": max_score,
                "question_text": "",
                "right_answer_text": "",
            })

    return pd.DataFrame(records)


def merge_grade_breakdown_rows(response_rows: pd.DataFrame, grade_rows: pd.DataFrame) -> pd.DataFrame:
    """Merge Grades-with-breakdown scores into response rows using student email and started-on timestamp."""
    if response_rows.empty:
        return response_rows.copy()
    if grade_rows.empty:
        return response_rows.copy()

    response_rows = response_rows.copy()
    grade_rows = grade_rows.copy()

    response_rows["started_on"] = response_rows.get("started_on", pd.NaT)
    response_rows["attempt_key"] = response_rows["student_id"].astype(str) + "|" + response_rows["started_on"].astype(str)
    grade_rows["attempt_key"] = grade_rows["student_id"].astype(str) + "|" + grade_rows["started_on"].astype(str)

    merged = response_rows.merge(
        grade_rows[["attempt_key", "question", "grade", "max_grade", "response_status", "response_text", "raw_score", "question_max_score", "source_type"]],
        on=["attempt_key", "question"],
        how="left",
        suffixes=("", "_grade"),
    )

    grade_mask = merged["grade_grade"].notna()
    if grade_mask.any():
        merged.loc[grade_mask, "grade"] = merged.loc[grade_mask, "grade_grade"]
    if "max_grade_grade" in merged.columns:
        merged.loc[merged["max_grade_grade"].notna(), "max_grade"] = merged.loc[merged["max_grade_grade"].notna(), "max_grade_grade"]
    if "response_status_grade" in merged.columns:
        merged.loc[merged["response_status_grade"].notna() & merged["response_status"].eq(""), "response_status"] = merged.loc[merged["response_status_grade"].notna() & merged["response_status"].eq(""), "response_status_grade"]
    if "source_type_grade" in merged.columns:
        merged.loc[merged["source_type_grade"].notna(), "source_type"] = merged.loc[merged["source_type_grade"].notna(), "source_type_grade"]
    if "raw_score_grade" in merged.columns:
        merged.loc[merged["raw_score_grade"].notna(), "raw_score"] = merged.loc[merged["raw_score_grade"].notna(), "raw_score_grade"]
    if "question_max_score_grade" in merged.columns:
        merged.loc[merged["question_max_score_grade"].notna(), "question_max_score"] = merged.loc[merged["question_max_score_grade"].notna(), "question_max_score_grade"]

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_grade")], errors="ignore")
    merged = merged.drop(columns=["attempt_key"], errors="ignore")
    return merged


def get_attempt_pools(response_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate response_df into (Pool A, Pool B).
    - Pool A ("All Attempts"): all Finished rows.
    - Pool B ("Best Attempt per Student"): exactly one attempt per student,
      selected as the row with the highest overall_grade (ties: latest completed_dt).
    """
    if response_df.empty or "attempt_idx" not in response_df.columns:
        return response_df, response_df

    pool_a_df = response_df.copy()

    # Find best attempt_idx per student
    attempt_meta = response_df[["student_id", "attempt_idx", "overall_grade", "completed_dt"]].drop_duplicates()
    attempt_meta = attempt_meta.sort_values(
        by=["overall_grade", "completed_dt", "attempt_idx"],
        ascending=[False, False, False]
    )
    best_attempts = attempt_meta.groupby("student_id").first().reset_index()
    best_indices = set(best_attempts["attempt_idx"])

    pool_b_df = response_df[response_df["attempt_idx"].isin(best_indices)].copy()
    return pool_a_df, pool_b_df

