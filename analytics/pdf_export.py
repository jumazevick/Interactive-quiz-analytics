from __future__ import annotations

import functools
import io
import logging
import os
import re
import tempfile
from typing import Any
import pandas as pd
import plotly.io as pio
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents

_logger = logging.getLogger(__name__)

_chrome_bootstrap_attempted = False


def _is_chrome_not_found(exc: Exception) -> bool:
    """kaleido >= 1.0 renders via a real headless Chrome instead of a bundled
    Chromium, and raises this specific error when none is discoverable on the host
    (e.g. a bare Streamlit Community Cloud container with no `chromium` apt package
    installed) — distinguishing it from other rasterization failures lets us attempt
    a one-time self-heal (_ensure_chrome_available) instead of just giving up."""
    try:
        from choreographer.errors import ChromeNotFoundError
        return isinstance(exc, ChromeNotFoundError)
    except Exception:
        return "chrome" in str(exc).lower() and "not" in str(exc).lower()


def _ensure_chrome_available() -> None:
    """One-time, best-effort attempt to download a private headless Chrome for
    kaleido when no system Chrome/Chromium was found. Safe to call repeatedly —
    only actually attempts the download once per process, and swallows failures
    (e.g. no outbound network access) since the caller already has a graceful
    fallback (the PDF simply notes the chart image is unavailable)."""
    global _chrome_bootstrap_attempted
    if _chrome_bootstrap_attempted:
        return
    _chrome_bootstrap_attempted = True
    try:
        import kaleido
        kaleido.get_chrome_sync()
    except Exception:
        _logger.warning("Could not download a private Chrome for kaleido chart export.", exc_info=True)


def _prepare_plotly_export(figure: Any) -> tuple[Any, int, int] | None:
    """Clone + re-style a Plotly figure for export (real template, legend, margins) —
    st.plotly_chart(...) themes figures on-screen via Streamlit's frontend at render
    time, which never happens when rasterizing server-side here, so a clone with an
    explicit template avoids silently losing color. Also forces a horizontal,
    below-plot legend and generous margins: a default right-side legend (esp. with
    many series, e.g. one per quiz) eats enough horizontal width at export time to
    squeeze the actual plot into a narrow, overlapping-label strip.

    Returns (prepared_figure, width, height), or None if `figure` isn't a Plotly
    figure at all (e.g. a Matplotlib figure or raw bytes, handled elsewhere).
    """
    if not hasattr(figure, "to_image"):
        return None
    export_width, export_height = 800, 400
    try:
        cloned = figure.__class__(figure)
        cloned.update_layout(
            template="plotly",
            margin=dict(l=50, r=50, t=50, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        )
        # Respect a figure's own explicit size (e.g. the student-performance heatmap
        # scales its height to the student count) instead of overriding it.
        if cloned.layout.width:
            export_width = cloned.layout.width
        if cloned.layout.height:
            export_height = cloned.layout.height
        return cloned, export_width, export_height
    except Exception:
        return figure, export_width, export_height


def _batch_rasterize_plotly_charts(sections: list[dict[str, Any]]) -> dict[int, bytes]:
    """Rasterize every Plotly figure across every section in ONE kaleido call instead
    of one call per chart.

    kaleido 1.x launches a fresh headless Chrome instance for each `fig.to_image()` /
    `write_image()` call (~4s of pure browser-startup overhead every time, confirmed
    by profiling — it does not get faster on repeat calls), which is what made PDF
    generation with several charts take tens of seconds. `plotly.io.write_images`
    (kaleido >= 1.0) amortizes that one Chrome startup across an entire batch: 10
    charts rasterize in ~4s total instead of ~40s. It writes to real file paths (an
    in-memory BytesIO target silently produces empty output), hence the temp dir.

    Returns {id(original_figure): png_bytes} for every figure that rasterized
    successfully; figures that fail or aren't Plotly figures are simply absent from
    the result, and the per-chart fallback path in generate_pdf_report handles them.
    """
    jobs: list[tuple[int, Any, int, int]] = []
    for sec in sections:
        for chart in (sec.get("charts") or []):
            chart_source = chart.get("figure", chart.get("image")) if isinstance(chart, dict) else chart
            prepared = _prepare_plotly_export(chart_source)
            if prepared is not None:
                fig, w, h = prepared
                jobs.append((id(chart_source), fig, w, h))

    if not jobs:
        return {}

    def _run_batch() -> dict[int, bytes]:
        batch_results: dict[int, bytes] = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [os.path.join(tmpdir, f"chart_{i}.png") for i in range(len(jobs))]
            pio.write_images(
                fig=[job[1] for job in jobs],
                file=paths,
                format="png",
                width=[job[2] for job in jobs],
                height=[job[3] for job in jobs],
                scale=2,
            )
            for (original_id, _, _, _), path in zip(jobs, paths):
                with open(path, "rb") as f:
                    data = f.read()
                if data:
                    batch_results[original_id] = data
        return batch_results

    try:
        return _run_batch()
    except Exception as exc:
        if _is_chrome_not_found(exc):
            _ensure_chrome_available()
            try:
                return _run_batch()
            except Exception:
                _logger.warning("Batch chart rasterization failed after Chrome bootstrap retry.", exc_info=True)
        else:
            _logger.warning("Batch chart rasterization failed.", exc_info=True)
        return {}  # Leave results empty; _figure_to_png_bytes below re-tries per chart.


def _figure_to_png_bytes(figure: Any) -> bytes | None:
    """Rasterize a single Plotly figure, a Matplotlib figure, or raw PNG bytes into PNG
    bytes. Used as the fallback for charts the batch pass in
    _batch_rasterize_plotly_charts didn't produce output for.

    Returns None (instead of raising) if rasterization fails — e.g. the optional
    kaleido package isn't installed — so a broken chart export can't take down the
    whole PDF (and the download button along with it).
    """
    if figure is None:
        return None
    if isinstance(figure, (bytes, bytearray)):
        return bytes(figure)

    prepared = _prepare_plotly_export(figure)
    if prepared is not None:
        fig, w, h = prepared
        try:
            return fig.to_image(format="png", width=w, height=h, scale=2)
        except Exception as exc:
            if _is_chrome_not_found(exc):
                _ensure_chrome_available()
                try:
                    return fig.to_image(format="png", width=w, height=h, scale=2)
                except Exception:
                    _logger.warning("Chart rasterization failed after Chrome bootstrap retry.", exc_info=True)
                    return None
            _logger.warning("Chart rasterization failed.", exc_info=True)
            return None

    if hasattr(figure, "savefig"):  # Matplotlib Figure
        try:
            buf = io.BytesIO()
            figure.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            _logger.warning("Matplotlib figure rasterization failed.", exc_info=True)
            return None
    return None


_MATH_DPI = 300
_MATH_PT_PER_PX = 72.0 / _MATH_DPI  # dpi is an exact px<->pt conversion, so every
# fragment rendered at the same requested font size comes back at the same physical
# size once converted back through this factor — normalizing to each fragment's own
# (glyph-shape-dependent) bounding-box height instead would make e.g. a plain "3/4"
# look bigger than a "sqrt(3)*i" answer rendered right next to it at the "same" size.


@functools.lru_cache(maxsize=4096)
def _render_math_png(fragment: str, fontsize: float) -> bytes | None:
    """Rasterize one line of text via Matplotlib's mathtext, which — unlike a plain
    reportlab Paragraph — natively supports mixed text/math strings (only the
    portions wrapped in `$...$` are typeset as math), so STACK answer expressions
    like `ans1: $-1/8$` render as actual fractions instead of literal dollar signs
    and backslash commands. Cached since the same right-answer/expression string
    repeats across many student rows in the drill-down tables.
    """
    if not fragment.strip():
        return None
    try:
        buf = io.BytesIO()
        math_to_image(fragment, buf, prop=FontProperties(size=fontsize), dpi=_MATH_DPI, color="#0f172a")
        return buf.getvalue()
    except Exception:
        return None


_LONG_FRAGMENT_THRESHOLD = 42
_MATH_SPAN_RE = re.compile(r"^(.*?)\$(.*)\$(.*)$", re.DOTALL)


def _wrap_long_math_fragment(fragment: str) -> list[str]:
    """A single `ansN: $...$` fragment can still be too wide to render at a legible
    size even after splitting on `; ` — e.g. a two-term trig identity like
    `9*i*sin(...)+9*cos(...)` — and since every fragment in a column shares one
    shrink factor (`_build_math_table_rows`), one such outlier drags down the whole
    column's font size. Break it at its top-level `+`/`-` (outside parens) onto
    additional lines, same as a human would wrap a long formula by hand, so the
    widest fragment in the column shrinks and the rest of the column can stay larger.
    """
    if len(fragment) <= _LONG_FRAGMENT_THRESHOLD or "$" not in fragment:
        return [fragment]
    match = _MATH_SPAN_RE.match(fragment)
    if not match or match.group(3).strip():
        return [fragment]
    prefix, inner, _ = match.groups()

    depth = 0
    split_points = []
    for i, ch in enumerate(inner):
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch in "+-" and depth == 0 and i > 0:
            split_points.append(i)
    if not split_points:
        return [fragment]

    parts = []
    start = 0
    for sp in split_points:
        parts.append(inner[start:sp])
        start = sp
    parts.append(inner[start:])
    return [f"{prefix if idx == 0 else ''}${part.strip()}$" for idx, part in enumerate(parts) if part.strip()]


def _split_math_fragments(text: str) -> list[str]:
    """Split cell text on newlines and on `; ` (the separator this codebase already
    uses between `ansN: ...` groups), so a long multi-answer cell stacks as several
    rasterized lines instead of being squeezed into one oversized, illegible image.
    Any fragment still too long gets further wrapped (see `_wrap_long_math_fragment`).
    """
    fragments = []
    for line in text.split("\n"):
        for fragment in line.split("; "):
            fragment = fragment.strip()
            if fragment:
                fragments.extend(_wrap_long_math_fragment(fragment))
    return fragments


def _build_math_table_rows(text_grid: list[list[str]], style: ParagraphStyle, col_widths: list[float]) -> list[list[Any]]:
    """Build reportlab table cell content for a whole df at once (rather than cell by
    cell), so that every math fragment in a given column shares ONE shrink-to-fit
    factor — the factor the column's single widest fragment needs. Scaling each
    fragment to fit independently (the original approach) made an `ans1: $-1/8$` cell
    look bigger than an `ans1: $9 \\cdot \\sin(...)$` cell right next to it, and even
    made `ans1`/`ans2` within the *same* cell inconsistent, since a longer expression
    needed more shrinking than a shorter one on the very next line.

    Cells with no `$...$` math pass through as a single Paragraph, unchanged from
    before. A fragment that fails to rasterize (e.g. an unsupported construct) falls
    back to plain Paragraph text rather than dropping the cell's content.
    """
    fontsize = style.fontSize * 1.35
    col_count = len(col_widths)

    # Pass 1: rasterize every math fragment once (cached across repeats — the same
    # right-answer string recurs across many student rows) and track, per column,
    # the widest natural (unshrunk) rendered width.
    cell_items: list[list[list[tuple[str, Any]]]] = []
    col_max_natural_width = [0.0] * col_count
    for row in text_grid:
        row_items = []
        for col_idx, val in enumerate(row):
            if not val:
                row_items.append([("text", "")])
                continue
            if "$" not in val:
                row_items.append([("text", val)])
                continue
            items: list[tuple[str, Any]] = []
            for fragment in _split_math_fragments(val):
                png_bytes = _render_math_png(fragment, fontsize)
                if not png_bytes:
                    items.append(("text", fragment))
                    continue
                reader = ImageReader(io.BytesIO(png_bytes))
                img_w, img_h = reader.getSize()
                draw_w, draw_h = img_w * _MATH_PT_PER_PX, img_h * _MATH_PT_PER_PX
                col_max_natural_width[col_idx] = max(col_max_natural_width[col_idx], draw_w)
                items.append(("math", (png_bytes, draw_w, draw_h)))
            row_items.append(items or [("text", "")])
        cell_items.append(row_items)

    col_shrink = [
        (col_widths[c] / col_max_natural_width[c]) if col_max_natural_width[c] > col_widths[c] else 1.0
        for c in range(col_count)
    ]

    # Pass 2: build the final flowables using each column's shared shrink factor.
    table_rows: list[list[Any]] = []
    for row_items in cell_items:
        row_cells = []
        for col_idx, items in enumerate(row_items):
            shrink = col_shrink[col_idx]
            flowables: list[Any] = []
            for kind, payload in items:
                if kind == "text":
                    flowables.append(Paragraph(payload, style))
                else:
                    png_bytes, draw_w, draw_h = payload
                    flowables.append(Image(io.BytesIO(png_bytes), width=draw_w * shrink, height=draw_h * shrink))
            row_cells.append(flowables)
        table_rows.append(row_cells)
    return table_rows


_NARROW_COLUMN_WEIGHTS = {
    "question": 0.55,
    "score": 0.45,
    "status": 0.6,
    "student name": 0.85,
    "frequency": 0.5,
}
_WIDE_COLUMN_KEYWORDS = ("response", "answer", "text", "email")


def _compute_column_widths(columns: list[str], usable_width: float) -> list[float]:
    """Weight columns by how much horizontal room their content actually needs,
    instead of splitting the page evenly — short fixed-vocabulary columns (Question,
    Score, Status, Student Name) get a smaller share so the free-text/math columns
    (Submitted Response, Right Answer, Most Common Incorrect Answer, ...) get more
    room and don't need to shrink their rendered math as aggressively.
    """
    weights = []
    for col in columns:
        key = str(col).strip().lower()
        if key in _NARROW_COLUMN_WEIGHTS:
            weights.append(_NARROW_COLUMN_WEIGHTS[key])
        elif any(keyword in key for keyword in _WIDE_COLUMN_KEYWORDS):
            weights.append(1.6)
        else:
            weights.append(1.0)
    total_weight = sum(weights) or 1.0
    return [usable_width * w / total_weight for w in weights]


class NumberedCanvas(canvas.Canvas):
    """Custom ReportLab canvas that adds a page header and 'Page X of Y' footer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header line & label
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, letter[1] - 40, letter[0] - 54, letter[1] - 40)
        self.drawString(54, letter[1] - 35, "Moodle STACK Analytics Hub — Performance Report")

        # Footer line & page numbers
        self.line(54, 50, letter[0] - 54, 50)
        self.drawString(54, 36, "Fully client-side export • No data transmitted externally")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, page_str)
        self.restoreState()


class _ReportDocTemplate(SimpleDocTemplate):
    """Feeds every section heading to the `TableOfContents` flowable and the PDF's
    native outline/bookmarks panel as it's laid out. Reportlab can't know a heading's
    page number until layout actually reaches it, so the auto-generated TOC (added at
    the front of the story) necessarily lags a build behind — `multiBuild` (used
    instead of `build` below) re-runs the layout until the TOC stops changing, which
    is reportlab's standard recipe for a self-populating table of contents.
    """

    def afterFlowable(self, flowable: Any) -> None:
        if isinstance(flowable, Paragraph) and getattr(flowable, "style", None) is not None and flowable.style.name == "SectionHeading":
            text = flowable.getPlainText()
            self.notify("TOCEntry", (0, text, self.page))
            key = f"section-{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=0, closed=True)


def generate_pdf_report(
    title: str,
    subtitle: str,
    sections: list[dict[str, Any]],
) -> bytes:
    """
    Generate a clean multi-page PDF report buffer.
    `sections` is a list of dicts:
      {
        "title": str,
        "caption": str,
        "df": pd.DataFrame,
        "charts": list[{"title": str, "figure": PlotlyFigure | MatplotlibFigure | bytes}],
        "notes": list[str]
      }
    """
    buffer = io.BytesIO()
    doc = _ReportDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=64,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e3c72"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=18,
    )

    section_heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=4,
    )

    caption_style = ParagraphStyle(
        "SectionCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=8,
    )

    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a"),
    )

    header_cell_style = ParagraphStyle(
        "TableHeaderCell",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
    )

    note_style = ParagraphStyle(
        "NoteText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
    )

    toc_heading_style = ParagraphStyle(
        "TOCHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#1e3c72"),
        spaceAfter=10,
    )

    story: list[Any] = []

    # Document Header
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(subtitle, subtitle_style))
    story.append(Spacer(1, 8))

    # Table of Contents — only worth the page when there's more than a couple of
    # sections to navigate; auto-populated by _ReportDocTemplate.afterFlowable as
    # layout reaches each "SectionHeading" (a distinct, non-"SectionHeading" style
    # here keeps this heading itself out of its own listing).
    if len(sections) > 2:
        toc = TableOfContents()
        toc.levelStyles = [
            ParagraphStyle(
                "TOCEntry",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=10,
                leading=14,
                textColor=colors.HexColor("#1e293b"),
            ),
        ]
        story.append(Paragraph("Table of Contents", toc_heading_style))
        story.append(toc)
        story.append(PageBreak())

    usable_width = letter[0] - 108  # 504 pt

    chart_png_cache = _batch_rasterize_plotly_charts(sections)

    for sec in sections:
        sec_title = sec.get("title", "")
        sec_caption = sec.get("caption", "")
        df = sec.get("df")
        notes = sec.get("notes") or []
        charts = sec.get("charts") or []

        # Heading + caption + table are kept together; chart images are appended as their
        # own flowables afterwards so a large table plus several charts can still flow
        # across page breaks instead of risking an oversized single KeepTogether block.
        header_elements: list[Any] = []
        if sec_title:
            header_elements.append(Paragraph(sec_title, section_heading_style))
        if sec_caption:
            header_elements.append(Paragraph(sec_caption, caption_style))

        if isinstance(df, pd.DataFrame) and not df.empty:
            col_widths = _compute_column_widths(list(df.columns), usable_width)
            # Default reportlab cell padding is 6pt each on left/right; give the math
            # rasterizer a little less than the raw column width so it shrinks to fit
            # inside the cell instead of touching the grid lines.
            cell_content_widths = [max(w - 12, 10) for w in col_widths]

            # Format headers
            headers = [Paragraph(str(col), header_cell_style) for col in df.columns]

            # Data rows
            df_slice = df.head(60)
            text_grid = [[str(v) if pd.notna(v) else "" for v in r] for _, r in df_slice.iterrows()]
            table_data = [headers] + _build_math_table_rows(text_grid, cell_style, cell_content_widths)

            t = Table(table_data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3c72")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ]))
            header_elements.append(t)
            header_elements.append(Spacer(1, 8))

        story.append(KeepTogether(header_elements))

        for chart in charts:
            if isinstance(chart, dict):
                chart_title = chart.get("title")
                chart_source = chart.get("figure", chart.get("image"))
            else:
                chart_title = None
                chart_source = chart

            png_bytes = chart_png_cache.get(id(chart_source)) or _figure_to_png_bytes(chart_source)
            if not png_bytes:
                if chart_title:
                    story.append(Paragraph(f"{chart_title} — chart image unavailable (rendering failed; check the app logs for details).", note_style))
                continue

            chart_elements: list[Any] = []
            if chart_title:
                chart_elements.append(Paragraph(chart_title, caption_style))

            image_reader = ImageReader(io.BytesIO(png_bytes))
            img_w, img_h = image_reader.getSize()
            max_height = 260
            scale = usable_width / img_w if img_w else 1.0
            draw_w, draw_h = img_w * scale, img_h * scale
            if draw_h > max_height:
                shrink = max_height / draw_h
                draw_w, draw_h = draw_w * shrink, draw_h * shrink

            chart_elements.append(Image(io.BytesIO(png_bytes), width=draw_w, height=draw_h))
            chart_elements.append(Spacer(1, 8))
            story.append(KeepTogether(chart_elements))

        note_elements: list[Any] = [Paragraph(f"• {note}", note_style) for note in notes]
        if note_elements:
            story.append(KeepTogether(note_elements))

        story.append(Spacer(1, 12))

    doc.multiBuild(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
