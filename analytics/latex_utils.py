from __future__ import annotations

import re


def clean_moodle_latex(text_str: str, is_header: bool = False) -> str:
    """Normalize raw STACK/Moodle LaTeX so st.markdown renders it as math instead of
    literal escape sequences or broken `$$` collisions.

    Moodle often emits adjacent inline math runs like `\\({3}\\)\\(\\,{-3}\\)` — naively
    replacing `\\(`/`\\)` with `$` independently collides the closing `$` of one run
    with the opening `$` of the next into `$$`, which Streamlit/KaTeX then treats as
    display math, breaking inline rendering and dropping brackets. Merging adjacent
    `\\)...\\(` pairs *before* converting delimiters avoids that collision at the source.

    `is_header=True` is for contexts that can't render multi-line display math or HTML
    (e.g. st.expander labels / table headers): display blocks collapse to inline `$...$`
    and newlines are flattened to spaces.
    """
    if not text_str:
        return text_str

    cleaned = text_str
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\\displaystyle\s*", "", cleaned)

    # 1. Merge adjacent inline math runs so `\)\(` can't collide into `$$` below.
    cleaned = re.sub(r"\\\)\s*\\\(", " ", cleaned)

    # 2. Display block math: `\[ ... \]` -> `$$ ... $$` (or `$ ... $` for headers).
    if is_header:
        cleaned = re.sub(r"\\\[(.*?)\\\]", r"$\1$", cleaned, flags=re.DOTALL)
    else:
        cleaned = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", cleaned, flags=re.DOTALL)

    # 3. Inline math: `\( ... \)` -> `$ ... $`.
    cleaned = re.sub(r"\\\((.*?)\\\)", r"$\1$", cleaned, flags=re.DOTALL)

    # 4. Clean up leftover collisions from adjacency types step 1 doesn't cover (e.g. a
    #    display block immediately followed by an inline run, `\]\(`, converts to a run
    #    of 3-4 `$`). A legitimate lone display-block boundary is exactly 2 `$` and must
    #    survive this — only collapse runs of 3 or more.
    cleaned = re.sub(r"(?:\$\s*){3,}", "$$", cleaned)

    if is_header:
        # Headers can't render multi-line display math or contain raw newlines.
        cleaned = cleaned.replace("$$", "$")
        cleaned = cleaned.replace("\n", " ")

    return re.sub(r"\s+", " ", cleaned).strip()


def _convert_sqrt(expr: str) -> str:
    """Balanced-paren replacement of Maxima `sqrt(...)` with LaTeX `\\sqrt{...}`,
    recursing into the inner content so nested sqrt()s convert correctly."""
    out = []
    i = 0
    while i < len(expr):
        if expr[i:i + 5] == "sqrt(":
            depth = 1
            j = i + 5
            start = j
            while j < len(expr) and depth > 0:
                if expr[j] == "(":
                    depth += 1
                elif expr[j] == ")":
                    depth -= 1
                j += 1
            inner = expr[start:j - 1]
            out.append(r"\sqrt{" + _convert_sqrt(inner) + "}")
            i = j
        else:
            out.append(expr[i])
            i += 1
    return "".join(out)


_FUNCTION_NAMES = ("sin", "cos", "tan", "asin", "acos", "atan", "log", "ln", "exp")


def maxima_expr_to_latex(expr: str) -> str:
    """Best-effort conversion of a Maxima/STACK CAS expression (e.g.
    `9*%i*sin((7*%pi)/12)+9*cos((7*%pi)/12)`) into LaTeX math, for display purposes
    only — not a full CAS parser, just the patterns STACK commonly emits."""
    if not expr:
        return expr
    out = expr
    out = out.replace("%i", r"\mathrm{i}")
    out = out.replace("%pi", r"\pi")
    out = _convert_sqrt(out)
    # Bare "sin"/"cos"/etc. render in math mode as italicized variable products
    # (s*i*n) unless flagged as LaTeX operator names.
    for name in _FUNCTION_NAMES:
        out = re.sub(rf"\b{name}\b", r"\\" + name, out)
    out = out.replace("*", r" \cdot ")
    return out


_ANS_PATTERN = re.compile(r"ans(\d+):\s*(.*?)\s*\[(score|valid|invalid)\]")


def extract_stack_answer_latex(raw_text: str) -> str:
    """Extract just the `ansN: <expression>` parts from a raw STACK response/right-answer
    dump (which also carries `Seed: ...` and `prtN: ...` diagnostic noise a teacher
    doesn't need), converting each expression to rendered LaTeX math wrapped in `$...$`
    — which also protects it from Markdown's emphasis parsing, the source of the random
    italics: a bare Maxima expression like `9*%i*sin(...)` has naked `*` characters that
    Markdown reads as *emphasis* markers when displayed unwrapped.

    Falls back to `clean_moodle_latex` on the whole string when no `ansN:` pattern is
    found, e.g. a plain (non-STACK) right-answer value.
    """
    if not raw_text:
        return raw_text

    matches = list(_ANS_PATTERN.finditer(raw_text))
    if not matches:
        return clean_moodle_latex(raw_text)

    parts = []
    for m in matches:
        idx, expr = m.group(1), m.group(2)
        parts.append(f"ans{idx}: ${maxima_expr_to_latex(expr)}$")
    return "; ".join(parts)
