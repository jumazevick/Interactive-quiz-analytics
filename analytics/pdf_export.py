from __future__ import annotations

import io
from typing import Any
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _figure_to_png_bytes(figure: Any) -> bytes | None:
    """Rasterize a Plotly figure, a Matplotlib figure, or raw PNG bytes into PNG bytes.

    Returns None (instead of raising) if rasterization fails — e.g. the optional
    kaleido package isn't installed — so a broken chart export can't take down the
    whole PDF (and the download button along with it).
    """
    if figure is None:
        return None
    if isinstance(figure, (bytes, bytearray)):
        return bytes(figure)
    try:
        if hasattr(figure, "to_image"):  # Plotly Figure
            # st.plotly_chart(...) themes charts on-screen via Streamlit's frontend at
            # render time; that color injection never happens when the same figure is
            # rasterized server-side here. Force a real (non-neutral) template on a
            # clone before exporting so PDF charts never silently lose their color,
            # regardless of what template the figure happened to pick up on-screen.
            # Also force a horizontal, below-plot legend and generous margins: a default
            # right-side legend (esp. with many series, e.g. one per quiz) eats enough
            # horizontal width at export time to squeeze the actual plot into a narrow,
            # overlapping-label strip.
            export_width, export_height = 800, 400
            try:
                cloned = figure.__class__(figure)
                cloned.update_layout(
                    template="plotly",
                    margin=dict(l=50, r=50, t=50, b=80),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                )
                # Respect a figure's own explicit size (e.g. the student-performance
                # heatmap scales its height to the student count) instead of overriding it.
                if cloned.layout.width:
                    export_width = cloned.layout.width
                if cloned.layout.height:
                    export_height = cloned.layout.height
                figure = cloned
            except Exception:
                pass
            return figure.to_image(format="png", width=export_width, height=export_height, scale=2)
        if hasattr(figure, "savefig"):  # Matplotlib Figure
            buf = io.BytesIO()
            figure.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            return buf.getvalue()
    except Exception:
        return None
    return None


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
    doc = SimpleDocTemplate(
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

    story: list[Any] = []

    # Document Header
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(subtitle, subtitle_style))
    story.append(Spacer(1, 8))

    usable_width = letter[0] - 108  # 504 pt

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
            table_data = []

            # Format headers
            headers = [Paragraph(str(col), header_cell_style) for col in df.columns]
            table_data.append(headers)

            # Data rows
            df_slice = df.head(60)
            for _, r in df_slice.iterrows():
                row_cells = []
                for val in r:
                    val_str = str(val) if pd.notna(val) else ""
                    row_cells.append(Paragraph(val_str, cell_style))
                table_data.append(row_cells)

            col_count = len(df.columns)
            col_width = usable_width / max(1, col_count)

            t = Table(table_data, colWidths=[col_width] * col_count)
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

            png_bytes = _figure_to_png_bytes(chart_source)
            if not png_bytes:
                if chart_title:
                    story.append(Paragraph(f"{chart_title} — chart image unavailable (kaleido not installed).", note_style))
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

    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
