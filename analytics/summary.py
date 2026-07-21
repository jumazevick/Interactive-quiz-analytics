import io
import pandas as pd


def build_export_summary(question_metrics: pd.DataFrame, response_outcomes: pd.DataFrame, difficulty_metrics: pd.DataFrame, syntax_analysis: pd.DataFrame, prt_pass_rates: pd.DataFrame, repeated_wrong_answers: pd.DataFrame) -> pd.DataFrame:
    """Create a combined summary table suitable for export."""
    summary = question_metrics.copy()
    if not response_outcomes.empty:
        summary = summary.merge(response_outcomes, on="question", how="left")
    if not difficulty_metrics.empty:
        summary = summary.merge(difficulty_metrics, on="question", how="left")
    if not syntax_analysis.empty:
        summary = summary.merge(syntax_analysis, on="question", how="left")
    if not repeated_wrong_answers.empty:
        summary = summary.rename(columns={"question": "question"})
        summary = summary.merge(repeated_wrong_answers, on="question", how="left")
    if not prt_pass_rates.empty:
        summary["prt_pass_count"] = int(prt_pass_rates.shape[0])
    return summary


def create_excel_report(data_dict: dict[str, pd.DataFrame]) -> bytes:
    """Create a multi-sheet Excel file from a dictionary of dataframes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()
