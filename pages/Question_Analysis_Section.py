import streamlit as st


st.set_page_config(
    page_title="Question Analysis Section",
    page_icon=":bar_chart:",
)

st.title("Question Analysis Section")
st.header("Question-level Analysis")

st.markdown(
    """
This section will contain the question-level analysis for Moodle STACK quiz results.
Upload a question results file below to get started. The analysis will be added here
once the analysis code is provided.
"""
)

uploaded_file = st.file_uploader(
    "Upload question results file",
    type=["csv", "xls", "xlsx"],
    help="Upload a Moodle question results export in CSV, XLS, or XLSX format.",
)

if uploaded_file is None:
    st.info("Upload a question results file to begin.")
else:
    st.success(f"File uploaded: {uploaded_file.name}")
    st.caption("Question analysis will appear here when the analysis code is added.")
