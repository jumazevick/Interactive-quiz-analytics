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


def build_response_rows(df: pd.DataFrame, quiz_name: str) -> pd.DataFrame:
    """Convert a Moodle export into a flattened question-response table."""
    if df.empty:
        return pd.DataFrame(columns=["student_id", "student_name", "question", "grade", "max_grade", "response_status", "response_text", "quiz_name"])

    normalized_df = normalize_question_columns(df)
    headers = list(normalized_df.columns)

    question_numbers = []
    for header in headers:
        match = re.match(r"^(?:Q|Question|q)\.?\s*(\d+)", header)
        if match:
            question_numbers.append(int(match.group(1)))
            continue
        match = re.match(r"^Response\s*(\d+)", header)
        if match:
            question_numbers.append(int(match.group(1)))

    question_numbers = sorted(set(question_numbers))
    records: list[dict[str, Any]] = []

    for index, row in normalized_df.iterrows():
        state_value = str(row.get("State", "")).strip().lower()
        if state_value and state_value != "finished":
            continue

        student_id = row.get("Email address") or row.get("anonymized_full_name") or f"student_{index}"
        student_name = f"{row.get('First name', '')} {row.get('Surname', '')}".strip() or "Anonymized Student"

        for question_number in question_numbers:
            question_label = f"Q{question_number}"
            response_text = ""
            response_col = None
            for header in headers:
                if re.match(rf"^response\s*0*{question_number}$", header, flags=re.IGNORECASE):
                    response_col = header
                    break
                if re.match(rf"^(?:Q|Question|q)\.?\s*0*{question_number}\s*response", header, flags=re.IGNORECASE):
                    response_col = header
                    break
                if re.match(rf"^q0*{question_number}_response", header, flags=re.IGNORECASE):
                    response_col = header
                    break
                if re.match(rf"^q0*{question_number}:response", header, flags=re.IGNORECASE):
                    response_col = header
                    break

            if response_col:
                response_text = str(row.get(response_col, "")).strip()

            grade_col = None
            for header in headers:
                if re.match(rf"^(?:Q|Question|q)\.?\s*0*{question_number}\s*(?:\/|$)", header, flags=re.IGNORECASE):
                    grade_col = header
                    break
                if re.match(rf"^(?:Q|Question|q)\.?\s*0*{question_number}_grade", header, flags=re.IGNORECASE):
                    grade_col = header
                    break
                if re.match(rf"^q0*{question_number}:grade", header, flags=re.IGNORECASE):
                    grade_col = header
                    break

            grade = 0.0
            max_grade = 1.0
            if grade_col:
                max_match = re.search(r"/(\d+(?:\.\d+)?)", grade_col)
                if max_match:
                    max_grade = float(max_match.group(1))
                grade_value = pd.to_numeric(row.get(grade_col), errors="coerce")
                if pd.notna(grade_value):
                    grade = float(grade_value)

            response_status = classify_response_status(response_text, grade, max_grade)

            records.append(
                {
                    "student_id": str(student_id),
                    "student_name": student_name,
                    "question": question_label,
                    "grade": grade,
                    "max_grade": max_grade,
                    "response_status": response_status,
                    "response_text": response_text,
                    "quiz_name": quiz_name,
                }
            )

    return pd.DataFrame(records)


def classify_response_status(response_text: str, grade: float, max_grade: float) -> str:
    """Classify a response into a coarse outcome category."""
    text = response_text.strip().lower()
    if "syntax" in text or "error" in text or response_text.strip() == "!":
        return "syntax_error"
    if "[invalid]" in text or "invalid" in text:
        return "invalid"
    if response_text.strip() == "":
        return "blank"
    if max_grade > 0 and grade >= max_grade * 0.9:
        return "correct"
    return "incorrect"
