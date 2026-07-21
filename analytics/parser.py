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


def build_response_rows(df: pd.DataFrame, quiz_name: str) -> pd.DataFrame:
    """Convert a Moodle export into a flattened question-response table."""
    # Row filtering: Drop any row where State is not exactly "Finished"
    state_cols = [col for col in df.columns if col.strip().lower() == "state"]
    if state_cols:
        df = df[df[state_cols[0]].astype(str).str.strip() == "Finished"]

    if df.empty:
        return pd.DataFrame(columns=[
            "student_id", "student_name", "question", "grade", "max_grade",
            "response_status", "response_text", "quiz_name", "ans_list",
            "prt_list", "overall_grade", "attempt_idx"
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
                "attempt_idx": index,
            })

    return pd.DataFrame(records)

