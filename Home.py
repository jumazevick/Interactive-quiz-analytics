import streamlit as st


st.set_page_config(page_title="Home", page_icon=":bar_chart:")

st.title("Home")
st.header("Moodle STACK Analytics Hub")
st.subheader("Streamlining Data Analysis for Moodle STACK")

st.page_link("pages/Quiz_Analysis_Section.py", label="Quiz Analysis Section")
st.page_link("pages/Question_Analysis_Section.py", label="Question Analysis Section")

st.markdown(
    """
This was a Hackathon project, thanks to Sage for helping with the technical setup.
It is still a work in progress. If you notice any bugs, please create a PR on GitHub.

Below is an intro video explaining the project idea:
"""
)

st.video("https://youtu.be/Ww_FrryExYc?si=x-yeDCqGgUhDjFMb")

st.markdown(
    """
**About this Tool**

This dashboard allows you to upload and analyze Moodle STACK quiz data:

- Supports `.csv`, `.xls`, `.xlsx` file formats
- Automatically detects and normalizes grades
- Parses quiz start and end times and time taken
- Visualizes quiz performance and trends

You need to upload your quiz files using the file uploader on the left side.
"""
)

st.markdown(
    """
---

### How to Export Quiz Attempt Data from Moodle

To use this dashboard, first download your Moodle quiz attempt data:

1. Log in to your Moodle course as a teacher or admin.
2. Navigate to the specific quiz you want to analyze.
3. Click on "Results" in the quiz menu.
4. Select "Grades" to see a list of attempts for all students.
5. Scroll to the bottom of the table and look for "Download table data as".
6. Choose `.CSV`, `.XLS`, or `.XLSX` and download the file.
7. Upload the file in the quiz analysis section.

### Required Columns Format

Make sure your file includes these columns exactly as named:

**Surname, First name, Email address, State, Started on, Completed, Time taken, Grade/10.00**

Moodle setups can vary, so if your columns are named differently, you may need to rename them before uploading.

Example files to test or adapt are available here:
[Sample Moodle Quiz Files](https://drive.google.com/drive/folders/1r7c1asoMFwaLORaQVKisJk7xpWazzC5I?usp=sharing)
"""
)
