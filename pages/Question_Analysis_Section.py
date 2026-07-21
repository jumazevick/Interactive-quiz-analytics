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

uploaded_file = st.sidebar.file_uploader(
    "Upload question results file",
    type=["csv", "xls", "xlsx"],
    help="Upload a Moodle question results export in CSV, XLS, or XLSX format.",
)

st.sidebar.title("Options")

placeholder_sections = {
    "Question Summary": "A summary of question-level performance will appear here.",
    "Question Difficulty Analysis": "Question difficulty and discrimination analysis will appear here.",
    "Question Response Distribution": "Response distributions for each question will appear here.",
    "Student Performance by Question": "Student performance comparisons by question will appear here.",
    "Question Metrics": "Additional question-level metrics and visualizations will appear here.",
}

if uploaded_file is None:
    st.info("Upload a question results file to begin.")
else:
    st.success(f"File uploaded: {uploaded_file.name}")

for section_name, placeholder_text in placeholder_sections.items():
    if st.sidebar.checkbox(section_name):
        st.subheader(section_name)
        st.info(placeholder_text)
        st.caption("Placeholder — analysis code will be added here.")
