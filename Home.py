import streamlit as st


st.set_page_config(page_title="Home", page_icon=":bar_chart:")

st.title("Interactive Quiz & Question Analytics")
st.subheader("Analyse Moodle STACK quiz results, calculate quiz and question-level statistics, explore Potential Response Trees (PRTs), and identify common student misconceptions.")

st.page_link("pages/Quiz_Analysis_Section.py", label="Quiz Analysis Section")
st.page_link("pages/Question_Analysis_Section.py", label="Question Analysis Section")

st.markdown("---")
st.markdown(
    """
### How to Export Quiz Attempt Data from Moodle

Follow these steps to download the datasets needed for the analytics workflow.

#### 1. Navigate to your quiz
Click into the specific STACK or standard Moodle quiz in your course workspace.

#### 2. Open quiz results
Select "Results" from the quiz secondary menu or settings menu.

#### 3. Choose a report type
Select either "Grades" or "Responses" depending on whether you want quiz-level or question-level analysis.

#### 4. Download table data
Scroll to the bottom of the page, choose Comma Separated Values (.csv) or Microsoft Excel (.xlsx), and click "Download".

### A. Quiz Analysis Only
Use the standard **Grades** export for overall grade trends, box plots, and scatter charts.
The downloaded spreadsheet should contain these columns:

- Surname
- First name
- Email address
- State
- Started on
- Completed
- Time taken
- Grade/10.00

### B. Question & PRT Analysis
Use the **Responses** export for question-level metrics, response distributions, and PRT analysis.
This report includes the same metadata columns above, plus a response column for each question item.

- Response 1
- Response 2
- Response 3
- ...
- Response N
"""
)

st.markdown("---")
st.markdown(
    """
### Sample Data
You can experiment with the provided sample exports to explore the dashboard before using your own Moodle exports.
"""
)

st.markdown("---")
st.markdown(
    """
### Project Information
This was originally developed as a Hackathon project.

Special thanks to:
- Sage for the technical setup.
- Ernest and Otis for the question analysis research and implementation.

The project is still a work in progress.
If you discover bugs or have suggestions for improvements, please open a Pull Request on GitHub.

Below is an introductory video explaining the project and its goals:
"""
)

st.video("https://youtu.be/Ww_FrryExYc?si=x-yeDCqGgUhDjFMb")
