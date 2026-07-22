from __future__ import annotations

import re
from typing import Any

import pandas as pd


def parse_uploaded_file(file_obj: Any) -> pd.DataFrame:
    """Load a Moodle export into a normalized DataFrame."""
    if file_obj.name.endswith(".xls"):
        df = pd.read_excel(file_obj, engine="xlrd")
    elif file_obj.name.endswith(".xlsx"):
        df = pd.read_excel(file_obj, engine="openpyxl")
    elif file_obj.name.endswith(".csv"):
        df = pd.read_csv(file_obj)
    else:
        raise ValueError(f"Unsupported file format: {file_obj.name}")

    if "Last name" in df.columns:
        df = df.rename(columns={"Last name": "Surname"})

    if "State" in df.columns and "state" not in df.columns:
        df = df.rename(columns={"State": "State"})

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
            "attempt_idx", "source_type"
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
        student_id = row.get("Email address") or row.get("anonymized_full_name") or f"student_{index}"
        first_name = row.get("First name", "")
        surname = row.get("Surname") or row.get("Last name") or ""
        student_name = f"{first_name} {surname}".strip() or "Anonymized Student"
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
            "raw_score", "question_max_score"
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
        student_id = row.get("Email address") or row.get("anonymized_full_name") or f"student_{index}"
        first_name = row.get("First name", "")
        surname = row.get("Surname") or row.get("Last name") or ""
        student_name = f"{first_name} {surname}".strip() or "Anonymized Student"
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

