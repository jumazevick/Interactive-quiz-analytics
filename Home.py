import re

import streamlit as st

st.set_page_config(page_title="Moodle/STACK Interactive Quiz Analytics", page_icon=":bar_chart:", layout="wide")


def render_youtube_video(url: str) -> None:
    """Normalize a YouTube URL (watch/youtu.be/embed, with or without tracking query
    params) into a form st.video() reliably embeds, with a graceful link fallback if
    the embed itself fails for any reason."""
    video_id = None
    for pattern in (
        r"youtu\.be/([A-Za-z0-9_-]{6,})",
        r"youtube\.com/watch\?v=([A-Za-z0-9_-]{6,})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{6,})",
    ):
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break

    try:
        st.video(f"https://www.youtube.com/watch?v={video_id}" if video_id else url)
    except Exception:
        st.warning("⚠️ The video couldn't be embedded here.")
        st.markdown(f"[Watch it directly on YouTube]({url})")

# Custom premium gradient header banner
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 2.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
        <span style="background-color: rgba(255, 255, 255, 0.2); color: white; padding: 4px 12px; border-radius: 16px; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; display: inline-block; margin-bottom: 10px;">
            Interactive STACK Data
        </span>
        <h1 style="color: white; margin: 0 0 10px 0; font-size: 2.5rem; font-weight: 800; border-bottom: none;">
            Moodle/STACK Interactive Quiz Analytics
        </h1>
        <p style="margin: 0; font-size: 1.1rem; opacity: 0.9; line-height: 1.5;">
            Analyze overall grade distributions, calculate correlation metrics, and drill down into specific student responses and potential response trees (PRTs) for STACK questions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Page Navigation link
st.page_link("pages/Question_and_Quiz_Analysis.py", label="📊 Go to Question & Quiz Analysis", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Export Steps container card
with st.container(border=True):
    st.markdown("### 📖 How to Export Quiz Attempt Data from Moodle")
    st.write("Follow these step-by-step instructions to download a compliant quiz attempt dataset from your Moodle course.")

    # Inner card for export steps
    with st.container(border=True):
        st.markdown("<h5 style='margin-top:0;'>⚙️ Moodle Export Steps</h5>", unsafe_allow_html=True)
        st.markdown(
            """
            1️⃣ **Navigate to your target Quiz** in Moodle.<br>
            2️⃣ Open **Quiz results**.<br>
            3️⃣ Select **Responses report** from the Moodle report dropdown menu.<br>
            4️⃣ Under **Display options**, check the boxes for: **Question text**, **Response**, and **Right answer**.<br>
            5️⃣ Click **Display report**.<br>
            6️⃣ Download the generated report as a **CSV** or **XLSX** file.<br>
            7️⃣ Verify that your file contains the required structure below.
            """,
            unsafe_allow_html=True,
        )

    # Inner card for required columns
    with st.container(border=True):
        st.markdown("### 📦 Expected Data Format (Columns from Left to Right)")
        st.write("Your uploaded CSV or XLSX file must contain column headers ordered sequentially across the table:")
        st.markdown(
            """
            **1. Columns 1 to 8 (Student & Quiz Metadata):**
            `Last name` | `First name` | `Email address` | `State` | `Started on` | `Completed` | `Time taken` | `Grade/10.00`

            **2. Columns 9+ (Repeating Question Triplets):**
            - `Question 1` | `Response 1` | `Right answer 1`
            - `Question 2` | `Response 2` | `Right answer 2`
            - ...
            - `Question N` | `Response N` | `Right answer N`
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
        st.link_button(
            "📥 Sample Quiz Files",
            url="https://drive.google.com/drive/folders/1r7c1asoMFwaLORaQVKisJk7xpWazzC5I?usp=sharing",
            use_container_width=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# Walkthrough Video Card
with st.container(border=True):
    st.markdown("### 🎥 Project Walkthrough Video")
    st.write("Below is an introductory video explaining the project and its goals:")
    render_youtube_video("https://youtu.be/Ww_FrryExYc?si=x-yeDCqGgUhDjFMb")

st.markdown("<br>", unsafe_allow_html=True)

# Project Info & Acknowledgements Card
with st.container(border=True):
    st.markdown("### 🤝 Project Information & Acknowledgements")
    st.write("This was originally developed as a Hackathon project.")
    st.markdown(
        """
        Special thanks to:
        - **Ernest** for the question analysis research, technical setup and development, and implementation.
        - **Sage** for the technical setup.
        - **Otis** for the question analysis research.
        """
    )
    st.write("The project is still a work in progress. If you discover bugs or have suggestions for improvements, please open a Pull Request on GitHub.")

# Persistent Footer (Part 6.2)
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: gray; margin-top: 1rem; margin-bottom: 1rem;">
        <div>Moodle/STACK Interactive Quiz Analytics is open-source and fully client-side.</div>
        <div>No quiz data is ever uploaded to external servers.</div>
    </div>
    """,
    unsafe_allow_html=True
)
