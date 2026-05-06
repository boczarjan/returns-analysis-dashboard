from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .metrics import aggregate_by, kpi_summary, product_ranking, reason_summary


INK = colors.HexColor("#102027")
GREEN = colors.HexColor("#31572c")
LIGHT_GREEN = colors.HexColor("#eaf6ec")
LINE = colors.HexColor("#cbd8cf")
MUTED = colors.HexColor("#63736d")


def _register_font() -> tuple[str, str]:
    regular_candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    bold_candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf"),
    ]

    regular = next((path for path in regular_candidates if path.exists()), None)
    bold = next((path for path in bold_candidates if path.exists()), None)
    if regular and bold:
        pdfmetrics.registerFont(TTFont("DashboardFont", str(regular)))
        pdfmetrics.registerFont(TTFont("DashboardFont-Bold", str(bold)))
        return "DashboardFont", "DashboardFont-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _register_font()


def _fmt_number(value: float) -> str:
    try:
        return f"{float(value):,.0f}".replace(",", " ")
    except (TypeError, ValueError):
        return ""


def _fmt_percent(value: float) -> str:
    try:
        return f"{float(value):.1f}%".replace(".", ",")
    except (TypeError, ValueError):
        return ""


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _paragraph(value: object, style: ParagraphStyle) -> Paragraph:
    escaped = _clean(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(escaped, style)


def _table(
    rows: Iterable[Iterable[object]],
    widths: list[float],
    body_style: ParagraphStyle,
    header_style: ParagraphStyle,
) -> Table:
    prepared = []
    for index, row in enumerate(rows):
        style = header_style if index == 0 else body_style
        prepared.append([_paragraph(cell, style) for cell in row])

    table = Table(prepared, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), GREEN),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("FONTNAME", (0, 1), (-1, -1), FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREEN]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _section(title: str, styles) -> list:
    return [Spacer(1, 0.25 * cm), Paragraph(title, styles["Section"]), Spacer(1, 0.12 * cm)]


def _on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.7 * cm, "Returns Analysis Dashboard")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.7 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf_report(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=1.15 * cm,
        rightMargin=1.15 * cm,
        topMargin=1 * cm,
        bottomMargin=1.15 * cm,
        title="Returns Analysis Report",
    )

    sample = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "Title",
            parent=sample["Title"],
            fontName=FONT_BOLD,
            fontSize=22,
            leading=26,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName=FONT,
            fontSize=9,
            leading=12,
            textColor=INK,
        ),
        "Muted": ParagraphStyle(
            "Muted",
            parent=sample["BodyText"],
            fontName=FONT,
            fontSize=8,
            leading=11,
            textColor=MUTED,
        ),
        "Section": ParagraphStyle(
            "Section",
            parent=sample["Heading2"],
            fontName=FONT_BOLD,
            fontSize=13,
            leading=16,
            textColor=GREEN,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "Header": ParagraphStyle(
            "Header",
            parent=sample["BodyText"],
            fontName=FONT_BOLD,
            fontSize=8,
            leading=10,
            textColor=colors.white,
        ),
        "Cell": ParagraphStyle(
            "Cell",
            parent=sample["BodyText"],
            fontName=FONT,
            fontSize=8,
            leading=10,
            textColor=INK,
        ),
    }

    summary = kpi_summary(df)
    story = [
        Paragraph("Returns Analysis Report", styles["Title"]),
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Rows after filters: {_fmt_number(len(df))}",
            styles["Muted"],
        ),
        Spacer(1, 0.25 * cm),
    ]

    kpi_rows = [
        ["Metric", "Value"],
        ["Sold articles", _fmt_number(summary["sold"])],
        ["Returned articles", _fmt_number(summary["returned"])],
        ["Weighted return rate", _fmt_percent(summary["return_rate"])],
        ["Average return rate", _fmt_percent(summary["average_return_rate"])],
        ["NMV", _fmt_number(summary["nmv"])],
        ["Article variants", _fmt_number(summary["variants"])],
    ]
    story += _section("KPI summary", styles)
    story.append(_table(kpi_rows, [5 * cm, 5 * cm], styles["Cell"], styles["Header"]))

    reasons = reason_summary(df).head(12)
    reason_rows = [["Reason", "Estimated returned articles", "Share of returns"]]
    for row in reasons.itertuples(index=False):
        reason_rows.append([row.reason, _fmt_number(row.estimated_returns), _fmt_percent(row.share_of_returns)])
    story += _section("Top return reasons", styles)
    story.append(_table(reason_rows, [9 * cm, 5 * cm, 4 * cm], styles["Cell"], styles["Header"]))

    countries = aggregate_by(df, "Country").head(14)
    country_rows = [["Country", "Sold", "Returned", "Return rate", "Return share"]]
    for row in countries.itertuples(index=False):
        country_rows.append(
            [
                row.Country,
                _fmt_number(row.sold),
                _fmt_number(row.returned),
                _fmt_percent(row.return_rate),
                _fmt_percent(row.return_share),
            ]
        )
    story += _section("Countries", styles)
    story.append(_table(country_rows, [3 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm], styles["Cell"], styles["Header"]))

    article_types = aggregate_by(df, "Article type").head(15)
    type_rows = [["Article type", "Sold", "Returned", "Return rate", "Variants"]]
    for row in article_types.itertuples(index=False):
        type_rows.append(
            [
                getattr(row, "_0", row[0]),
                _fmt_number(row.sold),
                _fmt_number(row.returned),
                _fmt_percent(row.return_rate),
                _fmt_number(row.variants),
            ]
        )
    story += _section("Article types", styles)
    story.append(_table(type_rows, [6 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm], styles["Cell"], styles["Header"]))

    products = product_ranking(df, min_sold=1).head(30)
    product_rows = [["Article variant", "Zalando variant", "Category", "Article type", "Sold", "Returned", "Return rate"]]
    for row in products.itertuples(index=False):
        product_rows.append(
            [
                row[0],
                row[1],
                row[2],
                row[3],
                _fmt_number(row.sold),
                _fmt_number(row.returned),
                _fmt_percent(row.return_rate),
            ]
        )
    story += _section("Top product variants", styles)
    story.append(
        _table(
            product_rows,
            [4.1 * cm, 4.1 * cm, 3 * cm, 4 * cm, 2.2 * cm, 2.5 * cm, 2.7 * cm],
            styles["Cell"],
            styles["Header"],
        )
    )

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    buffer.seek(0)
    return buffer.getvalue()
