import re

import streamlit as st

from analytics.ui_theme import inject_global_styles

st.set_page_config(page_title="Moodle/STACK Interactive Quiz Analytics", page_icon=":bar_chart:", layout="wide")
inject_global_styles()


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

# Header banner — monotone dark card (matches the dark theme in .streamlit/config.toml)
# with a single neutral accent badge, rather than a colored brand gradient.
st.markdown(
    """
    <div style="background: linear-gradient(180deg, #18181b 0%, #0f0f11 100%); border: 1px solid rgba(255, 255, 255, 0.08); padding: 2.5rem; border-radius: 12px; color: #f4f4f5; margin-bottom: 2rem;">
        <span style="background-color: rgba(255, 255, 255, 0.1); color: #e4e4e7; padding: 4px 12px; border-radius: 16px; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; display: inline-block; margin-bottom: 10px;">
            Interactive STACK Data
        </span>
        <h1 style="color: #f4f4f5; margin: 0 0 10px 0; font-size: 2.5rem; font-weight: 800; border-bottom: none;">
            Moodle/STACK Interactive Quiz Analytics
        </h1>
        <p style="margin: 0; font-size: 1.1rem; color: #a1a1aa; line-height: 1.5;">
            Analyze overall grade distributions, calculate correlation metrics, and drill down into specific student responses and potential response trees (PRTs) for STACK questions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Page Navigation link — styled as a prominent, high-contrast call-to-action button
# (light neutral fill against the dark banner, echoing the badge/text greys above).
st.markdown(
    """
    <style>
    [data-testid="stPageLink"] {
        background: #f4f4f5;
        border-radius: 10px;
        padding: 0.5rem 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stPageLink"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    }
    [data-testid="stPageLink"] p {
        color: #0a0a0b !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.page_link("pages/Question_and_Quiz_Analysis.py", label="📊 Go to Question & Quiz Analysis", use_container_width=True)

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
        Special Thanks and Credit to:
        - **Juma** for the original hackathon idea and implementation, quiz and question analysis research, and advising.
        - **Ernest** for the question analysis research, technical setup and development, UI improvements, and implementation.
        - **Sage Foundation** for the technical setup.
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
