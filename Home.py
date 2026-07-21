import streamlit as st

st.set_page_config(page_title="Home", page_icon=":bar_chart:", layout="wide")

# Custom premium gradient header banner
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 2.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
        <span style="background-color: rgba(255, 255, 255, 0.2); color: white; padding: 4px 12px; border-radius: 16px; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; display: inline-block; margin-bottom: 10px;">
            Interactive STACK Data
        </span>
        <h1 style="color: white; margin: 0 0 10px 0; font-size: 2.5rem; font-weight: 800; border-bottom: none;">
            Interactive Quiz & Question Analytics
        </h1>
        <p style="margin: 0; font-size: 1.1rem; opacity: 0.9; line-height: 1.5;">
            Analyze overall grade distributions, calculate correlation metrics, and drill down into specific student responses and potential response trees (PRTs) for STACK questions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Page Navigation links
col_link1, col_link2 = st.columns(2)
with col_link1:
    st.page_link("pages/Quiz_Analysis_Section.py", label="📈 Go to Quiz Analysis Section", use_container_width=True)
with col_link2:
    st.page_link("pages/Question_Analysis_Section.py", label="📊 Go to Question Analysis Section", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Export Steps container card
with st.container(border=True):
    st.markdown("### 📖 How to Export Quiz Attempt Data from Moodle")
    st.write("Follow these step-by-step instructions to download compliant quiz attempt datasets from your Moodle course. You can perform two types of analytics based on the reports you export.")
    
    # Inner card for general steps
    with st.container(border=True):
        st.markdown("<h5 style='margin-top:0;'>⚙️ General Export Steps</h5>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("1️⃣ **Navigate to Your Quiz**\n\nClick into the specific STACK or standard Moodle quiz in your course workspace.")
            st.markdown("2️⃣ **Open Quiz Results**\n\nSelect 'Results' from the quiz secondary menu or settings menu.")
        with col2:
            st.markdown("3️⃣ **Choose Report Type (Grades vs. Responses)**\n\nSelect either 'Grades' or 'Responses' from the Moodle report dropdown, depending on your target workflow.")
            st.markdown("4️⃣ **Download Table Data**\n\nScroll to the bottom of the page, select Comma Separated Values (.csv) or Microsoft Excel (.xlsx), and click 'Download'.")

st.markdown("<br>", unsafe_allow_html=True)

# Side-by-side export types cards
col_a, col_b = st.columns(2)

with col_a:
    with st.container(border=True):
        st.markdown("📝 **A. Quiz Analysis Only**")
        st.caption("For overall grade trends, box plots, and scatter charts")
        st.write("To evaluate overall quiz attempts, export the standard **Grades** report. The downloaded spreadsheet must contain exactly these columns:")
        with st.container(border=True):
            st.markdown(
                """
                - **Surname**
                - **First name**
                - **Email address**
                - **State**
                - **Started on**
                - **Completed**
                - **Time taken**
                - **Grade/10.00**
                """
            )

with col_b:
    with st.container(border=True):
        st.markdown("📦 **B. Question & PRT Analysis**")
        st.caption("For detailed question-level metrics and PRT trees")
        st.write("To evaluate specific question states and potential response trees (PRTs), export the **Responses** report. This process is identical to exporting grades, except you select 'Responses' instead of 'Grades'.")
        with st.container(border=True):
            st.markdown(
                """
                ```python
                // Identical metadata columns:
                - Surname, First name, Email address...
                - Grade/10.00
                // Plus response columns:
                - Response 1
                - Response 2
                - Response 3
                - ...
                - Response N (matches your quiz count)
                ```
                """
            )

st.markdown("<br>", unsafe_allow_html=True)

# Sample Data Download Section
with st.container(border=True):
    c_text, c_btn = st.columns([3, 1])
    with c_text:
        st.markdown("**Want to try some sample data?**")
        st.write("Download pre-configured mock quizzes and response reports to see the app in action.")
    with c_btn:
        zip_path = "sample_data/sample_quiz_data.zip"
        try:
            with open(zip_path, "rb") as f:
                zip_data = f.read()
            st.download_button(
                label="📥 Download Sample Quiz Files",
                data=zip_data,
                file_name="sample_quiz_data.zip",
                mime="application/zip",
                use_container_width=True
            )
        except FileNotFoundError:
            st.button("📥 Download Sample Quiz Files", disabled=True, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Walkthrough Video Card
with st.container(border=True):
    st.markdown("### 🎥 Project Walkthrough Video")
    st.write("Below is an introductory video explaining the project and its goals:")
    st.video("https://youtu.be/Ww_FrryExYc?si=x-yeDCqGgUhDjFMb")

st.markdown("<br>", unsafe_allow_html=True)

# Project Info & Acknowledgements Card
with st.container(border=True):
    st.markdown("### 🤝 Project Information & Acknowledgements")
    st.write("This was originally developed as a Hackathon project.")
    st.markdown(
        """
        Special thanks to:
        - **Sage** for the technical setup.
        - **Ernest** and **Otis** for the question analysis research and implementation.
        """
    )
    st.write("The project is still a work in progress. If you discover bugs or have suggestions for improvements, please open a Pull Request on GitHub.")

# Persistent Footer (Part 6.2)
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: gray; margin-top: 1rem; margin-bottom: 1rem;">
        <div>Moodle STACK Analytics Hub is open-source and fully client-side.</div>
        <div>No quiz data is ever uploaded to external servers.</div>
    </div>
    """,
    unsafe_allow_html=True
)
