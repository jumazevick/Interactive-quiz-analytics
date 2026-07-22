from __future__ import annotations

import re


def format_moodle_latex(text_str: str) -> str:
    """Normalize raw STACK/Moodle LaTeX so st.markdown renders it as math instead of
    literal escape sequences.

    - `\\(...\\)` (inline) -> `$...$`
    - `\\[...\\]` (display block) -> `$$...$$`
    - Strips stray HTML remnants (`<p>`, `<br>`, `<div>`) and the `\\displaystyle` marker
      that sometimes survive in question/response text.
    """
    if not text_str:
        return text_str

    cleaned = text_str
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)  # <p>, <br>, <div>, and any other stray tags
    cleaned = cleaned.replace(r"\displaystyle", "")

    # Display-block delimiters first, so `\(`/`\)` below doesn't also match inside them.
    cleaned = re.sub(r"\\\[", "$$", cleaned)
    cleaned = re.sub(r"\\\]", "$$", cleaned)
    cleaned = re.sub(r"\\\(", "$", cleaned)
    cleaned = re.sub(r"\\\)", "$", cleaned)

    return re.sub(r"\s+", " ", cleaned).strip()
