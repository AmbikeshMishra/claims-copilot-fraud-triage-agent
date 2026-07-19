"""Report node (brief §3d): one-page adjuster-style triage report —
extracted facts table, fired signals with evidence quotes, score, and a
recommended action tied to the triage bucket.

Purely deterministic formatting over already-computed state (no LLM call),
so it carries no throttle/cost-guard concerns.
"""

from datetime import datetime, timezone
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .schema import ExtractedClaim
from .signals import Signal

DISCLAIMER = (
    "Decision-support only, not an underwriting or denial system. A human adjuster makes the "
    "final call. All data shown is synthetic — no real claimants, policies, or medical records."
)

RECOMMENDED_ACTION = {
    "AUTO-APPROVE": (
        "No material fraud signals fired. Recommend proceeding with standard claim processing; "
        "no further review required at this time."
    ),
    "STANDARD REVIEW": (
        "One or more signals fired at moderate weight. Recommend a standard adjuster review of "
        "the flagged items below before approval."
    ),
    "INVESTIGATE": (
        "Multiple and/or high-weight signals fired. Recommend escalation to Special Investigation "
        "Unit (SIU) review before any payout decision."
    ),
}

BUCKET_COLORS = {
    "AUTO-APPROVE": colors.HexColor("#1e7e34"),
    "STANDARD REVIEW": colors.HexColor("#b3821b"),
    "INVESTIGATE": colors.HexColor("#c82333"),
}

FIELD_LABELS = {
    "claimant_name": "Claimant name",
    "claimant_id": "Claimant ID",
    "policy_no": "Policy number",
    "policy_start_date": "Policy start date",
    "date_of_loss": "Date of loss",
    "date_filed": "Date filed",
    "category": "Category",
    "amount_inr": "Amount claimed (INR)",
    "diagnosis": "Diagnosis",
    "treatment": "Treatment",
    "hospital": "Hospital",
    "narrative": "Narrative",
}


def _facts_rows(claim: ExtractedClaim) -> list[tuple[str, str]]:
    data = claim.model_dump()
    rows = []
    for key, label in FIELD_LABELS.items():
        value = data.get(key)
        if key == "amount_inr":
            value = f"Rs. {value:,}"
        rows.append((label, str(value) if value is not None else "—"))
    return rows


def _generated_at() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def build_report_markdown(
    claim_id: str,
    claim: ExtractedClaim,
    signals: list[Signal],
    score: int,
    bucket: str,
    borderline: bool = False,
    borderline_note: str | None = None,
) -> str:
    lines = [
        f"# Claims Triage Report — {claim_id}",
        "",
        f"*Generated {_generated_at()} — {DISCLAIMER}*",
        "",
        f"## Verdict: {bucket} (risk score {score}/100)",
        "",
    ]
    if borderline and borderline_note:
        lines += [f"⚠️ **Borderline verdict** — {borderline_note}", ""]
    lines += [
        f"**Recommended action:** {RECOMMENDED_ACTION[bucket]}",
        "",
        "## Extracted facts",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for label, value in _facts_rows(claim):
        value = value.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {label} | {value} |")

    lines += ["", "## Fraud signals", ""]
    for signal in signals:
        icon = "\U0001F6A9" if signal.fired else "✅"
        lines.append(f"- {icon} **{signal.name.replace('_', ' ')}** — {signal.explanation}")
        if signal.fired and signal.evidence:
            lines.append(f'  > "{signal.evidence}"')
    lines.append("")
    return "\n".join(lines)


def _pdf_styles() -> dict:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportTitle", parent=styles["Title"], fontSize=16, spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            "Disclaimer", parent=styles["Normal"], fontSize=7.5, textColor=colors.grey, spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHeading", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "SignalLine", parent=styles["Normal"], fontSize=9, spaceAfter=2, leading=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "Evidence", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"),
            leftIndent=14, spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Cell", parent=styles["Normal"], fontSize=8.5, leading=10.5,
        )
    )
    styles.add(
        ParagraphStyle(
            "Borderline", parent=styles["Normal"], fontSize=9, leading=12,
            textColor=colors.HexColor("#8a6100"), spaceBefore=4, spaceAfter=4,
        )
    )
    return styles


def build_report_pdf(
    claim_id: str,
    claim: ExtractedClaim,
    signals: list[Signal],
    score: int,
    bucket: str,
    borderline: bool = False,
    borderline_note: str | None = None,
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )
    styles = _pdf_styles()
    story = []

    story.append(Paragraph(f"Claims Triage Report — {claim_id}", styles["ReportTitle"]))
    story.append(Paragraph(f"Generated {_generated_at()} — {DISCLAIMER}", styles["Disclaimer"]))

    banner_color = BUCKET_COLORS.get(bucket, colors.grey)
    verdict_table = Table(
        [[Paragraph(f"<b>{bucket}</b> — risk score {score}/100", styles["Cell"])]],
        colWidths=[doc.width],
    )
    verdict_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), banner_color),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(verdict_table)
    story.append(Spacer(1, 4))
    if borderline and borderline_note:
        story.append(Paragraph(f"&#9888; <b>Borderline verdict</b> — {borderline_note}", styles["Borderline"]))
    story.append(Paragraph(f"<b>Recommended action:</b> {RECOMMENDED_ACTION[bucket]}", styles["SignalLine"]))

    story.append(Paragraph("Extracted facts", styles["SectionHeading"]))
    table_data = [[Paragraph("<b>Field</b>", styles["Cell"]), Paragraph("<b>Value</b>", styles["Cell"])]]
    for label, value in _facts_rows(claim):
        table_data.append([Paragraph(label, styles["Cell"]), Paragraph(value, styles["Cell"])])
    facts_table = Table(table_data, colWidths=[doc.width * 0.3, doc.width * 0.7])
    facts_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(facts_table)

    story.append(Paragraph("Fraud signals", styles["SectionHeading"]))
    for signal in signals:
        icon = "[FLAG]" if signal.fired else "[OK]"
        story.append(
            Paragraph(
                f"<b>{icon} {signal.name.replace('_', ' ')}</b> — {signal.explanation}",
                styles["SignalLine"],
            )
        )
        if signal.fired and signal.evidence:
            story.append(Paragraph(f'&ldquo;{signal.evidence}&rdquo;', styles["Evidence"]))

    doc.build(story)
    return buffer.getvalue()
