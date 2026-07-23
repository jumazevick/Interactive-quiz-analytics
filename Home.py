import re

import streamlit as st
import streamlit.components.v1 as components

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

# Header banner + page-link button colors follow the active theme (dark card /
# light button in dark mode, mirrored in light mode) instead of being hardcoded to
# one variant. st.context.theme.type only reflects the theme as of the *last Python
# rerun* — but picking System/Light/Dark from Streamlit's own "⋮" menu is a
# frontend-only change that never triggers a rerun (confirmed: stApp's
# data-test-script-state stays "notRunning" the whole time), so anything computed
# from it goes stale the instant the user toggles without also reloading the page.
# Native Streamlit widgets don't have this problem because their theme comes from a
# React context that re-renders live; our injected HTML has no such hook, so instead
# a tiny live watcher (below) observes the *actual rendered* background color of the
# app and flips a `data-app-theme` attribute on <body> the moment it changes — the
# CSS below keys off that attribute rather than off a value computed once in Python.
# The inline `is_dark_theme` guess still sets the very first paint (before the
# watcher's first tick, a few ms) so there's no initial flash of the wrong variant.
is_dark_theme = st.context.theme.type != "light"
_fallback_theme = "dark" if is_dark_theme else "light"

# Header banner — monotone card with a single neutral accent badge, rather than a
# colored brand gradient. Colors live entirely in the stylesheet below, scoped by
# `body[data-app-theme]`, so the watcher's attribute flip is all it takes to
# re-color everything — no per-element inline styles to keep in sync by hand.
st.markdown(
    f"""
    <style>
    .hero-banner {{ border-radius: 12px; padding: 2.5rem; margin-bottom: 2rem; border: 1px solid; }}
    .hero-badge {{ padding: 4px 12px; border-radius: 16px; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; display: inline-block; margin-bottom: 10px; }}
    .hero-title {{ margin: 0 0 10px 0; font-size: 2.5rem; font-weight: 800; border-bottom: none; }}
    .hero-subtitle {{ margin: 0; font-size: 1.1rem; line-height: 1.5; }}

    /* Fallback (server-guessed) variant, used only until the watcher script's first
    tick sets data-app-theme a few ms after paint — avoids a flash of unstyled
    content. The body[data-app-theme] rules below are meant to always win once that
    attribute is actually set, so they're !important: a plain specificity fight
    isn't reliable here since the fallback selectors necessarily chain through
    .hero-banner to scope themselves, which — for the nested badge/title/subtitle —
    makes them MORE specific than "body[data-app-theme=...] .hero-title" despite
    being the one that should lose (found by testing the live toggle, not by
    inspection: the title silently kept the fallback's color after the attribute
    flipped). */
    .hero-banner[data-app-theme-fallback="dark"] {{ background: linear-gradient(180deg, #18181b 0%, #0f0f11 100%); border-color: rgba(255, 255, 255, 0.08); color: #f4f4f5; }}
    .hero-banner[data-app-theme-fallback="dark"] .hero-badge {{ background-color: rgba(255, 255, 255, 0.1); color: #e4e4e7; }}
    .hero-banner[data-app-theme-fallback="dark"] .hero-title {{ color: #f4f4f5; }}
    .hero-banner[data-app-theme-fallback="dark"] .hero-subtitle {{ color: #a1a1aa; }}
    .hero-banner[data-app-theme-fallback="light"] {{ background: linear-gradient(180deg, #f4f4f5 0%, #e4e4e7 100%); border-color: rgba(0, 0, 0, 0.08); color: #18181b; }}
    .hero-banner[data-app-theme-fallback="light"] .hero-badge {{ background-color: rgba(0, 0, 0, 0.08); color: #3f3f46; }}
    .hero-banner[data-app-theme-fallback="light"] .hero-title {{ color: #18181b; }}
    .hero-banner[data-app-theme-fallback="light"] .hero-subtitle {{ color: #52525b; }}

    body[data-app-theme="dark"] .hero-banner {{ background: linear-gradient(180deg, #18181b 0%, #0f0f11 100%) !important; border-color: rgba(255, 255, 255, 0.08) !important; color: #f4f4f5 !important; }}
    body[data-app-theme="dark"] .hero-badge {{ background-color: rgba(255, 255, 255, 0.1) !important; color: #e4e4e7 !important; }}
    body[data-app-theme="dark"] .hero-title {{ color: #f4f4f5 !important; }}
    body[data-app-theme="dark"] .hero-subtitle {{ color: #a1a1aa !important; }}

    body[data-app-theme="light"] .hero-banner {{ background: linear-gradient(180deg, #f4f4f5 0%, #e4e4e7 100%) !important; border-color: rgba(0, 0, 0, 0.08) !important; color: #18181b !important; }}
    body[data-app-theme="light"] .hero-badge {{ background-color: rgba(0, 0, 0, 0.08) !important; color: #3f3f46 !important; }}
    body[data-app-theme="light"] .hero-title {{ color: #18181b !important; }}
    body[data-app-theme="light"] .hero-subtitle {{ color: #52525b !important; }}
    </style>
    <div class="hero-banner" data-app-theme-fallback="{_fallback_theme}">
        <span class="hero-badge">Interactive STACK Data</span>
        <h1 class="hero-title">Moodle/STACK Interactive Quiz Analytics</h1>
        <p class="hero-subtitle">
            Analyze overall grade distributions, calculate correlation metrics, and drill down into specific student responses and potential response trees (PRTs) for STACK questions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Page Navigation link — styled as a prominent, high-contrast call-to-action button
# (light neutral fill against the dark banner, echoing the badge/text greys above —
# and vice versa in light mode). Same data-app-theme-driven approach as the banner;
# the unscoped rule below is the fallback (server-guessed) variant, same reasoning
# as the banner's, and is automatically outweighed by the body[data-app-theme] rules
# once the watcher script sets that attribute (an ancestor+descendant selector is
# more specific than the bare descendant selector alone).
_fallback_button_bg = "#f4f4f5" if is_dark_theme else "#18181b"
_fallback_button_text = "#0a0a0b" if is_dark_theme else "#f4f4f5"
_fallback_button_shadow = "rgba(0, 0, 0, 0.25)" if is_dark_theme else "rgba(0, 0, 0, 0.15)"
_fallback_button_shadow_hover = "rgba(0, 0, 0, 0.35)" if is_dark_theme else "rgba(0, 0, 0, 0.25)"
st.markdown(
    f"""
    <style>
    [data-testid="stPageLink"] {{
        background: {_fallback_button_bg};
        border-radius: 10px;
        padding: 0.5rem 0.5rem;
        box-shadow: 0 2px 8px {_fallback_button_shadow};
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    [data-testid="stPageLink"]:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 14px {_fallback_button_shadow_hover};
    }}
    [data-testid="stPageLink"] p {{
        color: {_fallback_button_text} !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
    }}
    body[data-app-theme="dark"] [data-testid="stPageLink"] {{ background: #f4f4f5 !important; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important; }}
    body[data-app-theme="dark"] [data-testid="stPageLink"]:hover {{ box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important; }}
    body[data-app-theme="dark"] [data-testid="stPageLink"] p {{ color: #0a0a0b !important; }}

    body[data-app-theme="light"] [data-testid="stPageLink"] {{ background: #18181b !important; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important; }}
    body[data-app-theme="light"] [data-testid="stPageLink"]:hover {{ box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important; }}
    body[data-app-theme="light"] [data-testid="stPageLink"] p {{ color: #f4f4f5 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)
st.page_link("pages/Question_and_Quiz_Analysis.py", label="📊 Go to Question & Quiz Analysis", use_container_width=True)

# Live theme watcher: st.markdown's HTML is inserted via innerHTML, so a plain
# <script> tag in it would never execute (a standard browser security restriction,
# not a Streamlit limitation) — st.components.v1.html runs in a real iframe where
# scripts do execute, and same-origin access lets it reach `window.parent.document`
# to actually observe and patch the outer app page. It infers dark-vs-light from the
# app's own rendered background color (luminance) rather than any Streamlit-internal
# API, since nothing else observably reflects the "⋮" menu's theme choice.
components.html(
    """
    <script>
    (function() {
        function isDarkBackground(el) {
            var bg = window.parent.getComputedStyle(el).backgroundColor;
            var m = bg.match(/\\d+/g);
            if (!m || m.length < 3) return null;
            var r = parseInt(m[0], 10), g = parseInt(m[1], 10), b = parseInt(m[2], 10);
            return (0.299 * r + 0.587 * g + 0.114 * b) < 128;
        }
        var observedApp = null;
        function applyTheme() {
            var appEl = window.parent.document.querySelector('[data-testid="stApp"]');
            if (!appEl) return;
            var isDark = isDarkBackground(appEl);
            if (isDark === null) return;
            window.parent.document.body.setAttribute('data-app-theme', isDark ? 'dark' : 'light');
            if (appEl !== observedApp) {
                observedApp = appEl;
                new MutationObserver(applyTheme).observe(appEl, {attributes: true, attributeFilter: ['class']});
            }
        }
        applyTheme();
        // Safety net beyond the MutationObserver, in case a future Streamlit version
        // re-themes stApp without a class-attribute mutation, or replaces the node
        // entirely (e.g. across a page navigation) before the observer re-attaches.
        setInterval(applyTheme, 800);
    })();
    </script>
    """,
    height=0,
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
        Special thanks and credit to:
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
