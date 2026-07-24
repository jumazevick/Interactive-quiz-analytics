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


def _convert_pow_parens(expr: str) -> str:
    """Balanced-paren replacement of Maxima `^(...)` with LaTeX `^{...}`.

    LaTeX superscripts only the single token immediately after `^`, so a raw
    `e^(pi*x)` renders as `e` to the power of just `(` — the rest of the exponent
    (`pi*x)`) drops back to the baseline instead of being raised. Wrapping the
    parenthesized exponent in braces (recursing for nested parens) fixes that."""
    out = []
    i = 0
    while i < len(expr):
        if expr[i] == "^" and expr[i + 1:i + 2] == "(":
            depth = 1
            j = i + 2
            start = j
            while j < len(expr) and depth > 0:
                if expr[j] == "(":
                    depth += 1
                elif expr[j] == ")":
                    depth -= 1
                j += 1
            inner = expr[start:j - 1]
            out.append("^{" + _convert_pow_parens(inner) + "}")
            i = j
        else:
            out.append(expr[i])
            i += 1
    return "".join(out)


def _convert_bracket_call(expr: str, func_name: str, open_wrap: str, close_wrap: str) -> str:
    """Balanced-paren replacement of a Maxima `func_name(...)` call (e.g. `abs(...)`,
    `floor(...)`) with `open_wrap ... close_wrap`, recursing into the inner content so
    nested calls of the same function convert correctly."""
    marker = func_name + "("
    out = []
    i = 0
    n = len(expr)
    while i < n:
        if expr[i:i + len(marker)] == marker:
            depth = 1
            j = i + len(marker)
            start = j
            while j < n and depth > 0:
                if expr[j] == "(":
                    depth += 1
                elif expr[j] == ")":
                    depth -= 1
                j += 1
            inner = expr[start:j - 1]
            out.append(open_wrap + _convert_bracket_call(inner, func_name, open_wrap, close_wrap) + close_wrap)
            i = j
        else:
            out.append(expr[i])
            i += 1
    return "".join(out)


def _convert_nthroot(expr: str) -> str:
    """Balanced-paren replacement of Maxima `nthroot(x, n)` with LaTeX `\\sqrt[n]{x}`."""
    marker = "nthroot("
    out = []
    i = 0
    n = len(expr)
    while i < len(expr):
        if expr[i:i + len(marker)] == marker:
            depth = 1
            j = i + len(marker)
            start = j
            while j < n and depth > 0:
                if expr[j] == "(":
                    depth += 1
                elif expr[j] == ")":
                    depth -= 1
                j += 1
            inner = expr[start:j - 1]
            args = _split_top_level(inner, ",")
            if len(args) == 2:
                value, degree = args
                out.append(r"\sqrt[" + _convert_nthroot(degree.strip()) + "]{" + _convert_nthroot(value.strip()) + "}")
            else:
                # Not the expected 2-argument shape — leave it for the generic
                # operatorname fallback rather than guessing at a malformed call.
                out.append(marker + _convert_nthroot(inner) + ")")
            i = j
        else:
            out.append(expr[i])
            i += 1
    return "".join(out)


# Maxima/STACK function names that map onto a *standard* LaTeX operator macro
# (confirmed to render in both KaTeX, used on-screen, and Matplotlib's mathtext,
# used for the PDF export — neither supports arbitrary `\operatorname`-free macros).
#
# Deliberately excludes names that double as plausible plain variable names in
# this quiz domain (`min`, `max`, `det`, `gcd`, `lim`, `sup`, `arg`, `deg`, ...) —
# unlike this list (matched unconditionally, call or not), those are left for the
# `_GENERIC_CALL_RE` fallback below, which only fires when the name is actually
# *called* (immediately followed by `(`), so a bare variable named e.g. `min`
# renders unchanged instead of being forced into operator form.
_FUNCTION_NAMES = (
    "sin", "cos", "tan", "sinh", "cosh", "tanh", "sec", "csc", "cot",
    "arcsin", "arccos", "arctan",
    "log", "ln", "exp",
)

# Maxima spells the inverse trig functions without the "arc" prefix; `\asin`/`\acos`/
# `\atan` are not valid LaTeX macros in either renderer, so rename to the standard
# `arcsin`/`arccos`/`arctan` spelling before the backslash-prefixing pass below.
_RENAMED_FUNCTIONS = {"asin": "arcsin", "acos": "arccos", "atan": "arctan"}

# Bare Maxima constants/keywords that read as nonsense if left as literal math-mode
# variables (e.g. "true" would otherwise render as the product t*r*u*e).
_WORD_REPLACEMENTS = {
    "true": r"\text{true}",
    "false": r"\text{false}",
    "unknown": r"\text{unknown}",
    "infinity": r"\infty",
    "minf": r"-\infty",
    "inf": r"\infty",
    "und": r"\text{undefined}",
    "ind": r"\text{indeterminate}",
}
_WORD_REPLACEMENT_RE = re.compile(
    r"\b(" + "|".join(sorted(_WORD_REPLACEMENTS, key=len, reverse=True)) + r")\b"
)

# A bare identifier directly followed by "(" is, in Maxima's plain-text output,
# always a function call (Maxima always writes multiplication with an explicit
# `*`, never juxtaposition) — so anything not already handled above is wrapped in
# `\operatorname{}` as a catch-all. This is what lets arbitrary/uncommon Maxima
# functions (`binomial(...)`, `mod(...)`, `sum(...)`, a custom teacher function,
# etc.) still render legibly instead of as italicized, implicitly-multiplied
# letters. Skips names already backslash-prefixed by an earlier pass.
_GENERIC_CALL_RE = re.compile(r"(?<!\\)\b([A-Za-z_][A-Za-z0-9_]*)\(")


def _convert_term(expr: str) -> str:
    """Convert a Maxima expression that does not itself contain a `matrix(...)` call."""
    out = expr
    out = out.replace("%i", r"\mathrm{i}")
    out = out.replace("%pi", r"\pi")
    out = out.replace("%e", r"\mathrm{e}")
    out = out.replace("%phi", r"\varphi")
    out = out.replace("%gamma", r"\gamma")
    # Any remaining bare `%` (an unhandled Maxima constant/label like `%o1`) is a LaTeX
    # comment marker to KaTeX and silently truncates everything after it when rendered
    # unescaped — escape defensively instead of losing the rest of the expression.
    out = out.replace("%", r"\%")

    out = _WORD_REPLACEMENT_RE.sub(lambda m: _WORD_REPLACEMENTS[m.group(1)], out)
    # Maxima's not-equal operator; comparisons/inequalities read better with the
    # proper math symbols than the plain ASCII Maxima emits.
    out = out.replace("#", r" \neq ")
    out = out.replace("<=", r" \leq ")
    out = out.replace(">=", r" \geq ")

    out = _convert_sqrt(out)
    out = _convert_nthroot(out)
    out = _convert_bracket_call(out, "abs", "|", "|")
    out = _convert_bracket_call(out, "floor", r"\lfloor ", r" \rfloor")
    out = _convert_bracket_call(out, "ceiling", r"\lceil ", r" \rceil")
    out = _convert_pow_parens(out)

    for src, dst in _RENAMED_FUNCTIONS.items():
        out = re.sub(rf"\b{src}\b", dst, out)

    # Bare "sin"/"cos"/etc. render in math mode as italicized variable products
    # (s*i*n) unless flagged as LaTeX operator names.
    for name in _FUNCTION_NAMES:
        out = re.sub(rf"\b{name}\b", r"\\" + name, out)

    # Catch-all for any other Maxima function call left unconverted above.
    out = _GENERIC_CALL_RE.sub(r"\\operatorname{\1}(", out)

    out = out.replace("*", r" \cdot ")
    return out


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split on `sep` only where it isn't nested inside `()`/`[]`/`{}` — a plain `.split`
    would break a matrix row like `[1,2]` (or a function-call element `sqrt(2,3)`) apart
    at the wrong comma."""
    parts = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _convert_matrix(expr: str) -> str:
    """Convert every Maxima `matrix([r1c1,r1c2,...],[r2c1,...],...)` call in `expr` into
    a LaTeX matrix, converting each cell (and any text outside the matrix call) through
    `_convert_term` so `%pi`/`sqrt(...)`/etc. inside a matrix cell still render correctly.

    Deliberately avoids `\\begin{bmatrix}...\\end{bmatrix}`: this text is rendered in two
    places — Streamlit/KaTeX on-screen, and Matplotlib's `mathtext` for the PDF export —
    and mathtext has no support for LaTeX environments at all (`\\begin{...}` fails to
    parse outright), so a bmatrix would render fine on-screen but fall back to raw,
    broken text in the PDF. `\\left[\\substack{row \\\\ row}\\right]`, with row entries
    separated by `\\quad`, is supported by both and gives an equivalent bracketed,
    stacked layout (columns aren't grid-aligned across rows, since `\\substack` centers
    each line independently, but that's a minor cosmetic tradeoff for working in both
    renderers)."""
    out = []
    buffer: list[str] = []
    i = 0
    n = len(expr)
    while i < n:
        if expr[i:i + 7] == "matrix(":
            if buffer:
                out.append(_convert_term("".join(buffer)))
                buffer = []
            depth = 1
            j = i + 7
            start = j
            while j < n and depth > 0:
                if expr[j] == "(":
                    depth += 1
                elif expr[j] == ")":
                    depth -= 1
                j += 1
            inner = expr[start:j - 1]

            latex_rows = []
            for row in _split_top_level(inner, ","):
                row = row.strip()
                if row.startswith("[") and row.endswith("]"):
                    row = row[1:-1]
                elements = [_convert_term(e.strip()) for e in _split_top_level(row, ",")]
                latex_rows.append(r" \quad ".join(elements))

            out.append(r"\left[\substack{" + r" \\ ".join(latex_rows) + r"}\right]")
            i = j
        else:
            buffer.append(expr[i])
            i += 1
    if buffer:
        out.append(_convert_term("".join(buffer)))
    return "".join(out)


def maxima_expr_to_latex(expr: str) -> str:
    """Best-effort conversion of a Maxima/STACK CAS expression (e.g.
    `9*%i*sin((7*%pi)/12)+9*cos((7*%pi)/12)`, or a matrix like `matrix([8],[24],[20])`)
    into LaTeX math, for display purposes only — not a full CAS parser, just the
    patterns STACK commonly emits."""
    if not expr:
        return expr
    if "matrix(" in expr:
        return _convert_matrix(expr)
    return _convert_term(expr)


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
