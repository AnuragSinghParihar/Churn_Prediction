"""
export.py — PDF report generation for the engagement analysis output.

Uses reportlab (a pure-Python PDF library). Install with:
    pip install reportlab

If reportlab is not installed, REPORTLAB_AVAILABLE is False and the
Streamlit page will show an install hint instead of the download button.
"""
from __future__ import annotations

import io
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        HRFlowable,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

_DARK = "#2c3e50"
_ACCENT = "#e74c3c"
_LIGHT_GREY = "#f5f5f5"


def export_pdf(report: dict, player_data: dict) -> bytes:
    """Build a formatted PDF engagement report and return it as bytes.

    Parameters
    ----------
    report : dict
        Structured output from EngagementAgent.run().
    player_data : dict
        Original player feature values.

    Returns
    -------
    bytes
        PDF file content (suitable for st.download_button).

    Raises
    ------
    ImportError
        If reportlab is not installed.
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab not installed. Run: pip install reportlab"
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    # ── Heading styles ────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor(_DARK),
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor(_DARK),
        spaceBefore=12,
        spaceAfter=4,
    )
    body_style = styles["Normal"]
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        spaceBefore=4,
    )

    # ── Title block ───────────────────────────────────────────────────────────
    story.append(Paragraph("Player Engagement Analysis Report", title_style))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
            f"Model: Churn Prediction System",
            caption_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor(_ACCENT)))
    story.append(Spacer(1, 0.4 * cm))

    # ── Player Profile table ──────────────────────────────────────────────────
    story.append(Paragraph("Player Profile", h2_style))

    table_data = [["Feature", "Value"]]
    for k, v in player_data.items():
        table_data.append([str(k), str(v)])

    profile_table = Table(table_data, colWidths=[8 * cm, 8 * cm])
    profile_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_DARK)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(_LIGHT_GREY)]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ]
        )
    )
    story.append(profile_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── Text sections ─────────────────────────────────────────────────────────
    text_sections = [
        ("Behaviour Summary", "player_behavior_summary"),
        ("Churn Risk Interpretation", "churn_risk_interpretation"),
        ("User Experience Notes", "user_experience_notes"),
    ]
    for heading, key in text_sections:
        content = report.get(key, "").strip()
        if content:
            story.append(Paragraph(heading, h2_style))
            story.append(Paragraph(content, body_style))

    # ── Recommendations ───────────────────────────────────────────────────────
    recs = report.get("engagement_recommendations", [])
    if recs:
        story.append(Paragraph("Engagement Recommendations", h2_style))
        for i, rec in enumerate(recs, 1):
            story.append(Paragraph(f"{i}.  {rec}", body_style))
            story.append(Spacer(1, 0.15 * cm))

    # ── Supporting references ─────────────────────────────────────────────────
    refs = report.get("supporting_references", [])
    if refs:
        story.append(Paragraph("Supporting References", h2_style))
        for ref in refs:
            story.append(Paragraph(f"\u2022  {ref}", body_style))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    disclaimer = report.get("ethical_disclaimer", "")
    if disclaimer:
        story.append(Paragraph(f"\u26a0  Ethical Note: {disclaimer}", caption_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
