"""Render data/claims_source.json into one claim-form PDF per claim in data/claims_pdfs/.

Usage:
    python scripts/generate_claim_pdfs.py
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "data" / "claims_source.json"
OUTPUT_DIR = ROOT / "data" / "claims_pdfs"

styles = getSampleStyleSheet()
title_style = ParagraphStyle("ClaimTitle", parent=styles["Heading1"], fontSize=14, spaceAfter=4)
insurer_style = ParagraphStyle("Insurer", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#444444"))
disclaimer_style = ParagraphStyle(
    "Disclaimer",
    parent=styles["Normal"],
    fontSize=8,
    textColor=colors.HexColor("#B00020"),
    spaceBefore=2,
    spaceAfter=10,
)
section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=11, spaceBefore=12, spaceAfter=4)
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)
footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#888888"))
cell_label_style = ParagraphStyle("CellLabel", parent=styles["Normal"], fontSize=9, leading=11, fontName="Helvetica-Bold")
cell_value_style = ParagraphStyle("CellValue", parent=styles["Normal"], fontSize=9, leading=11)

TABLE_STYLE = TableStyle(
    [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F2F2")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
)


def format_date(iso_date: str) -> str:
    from datetime import datetime

    return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d-%b-%Y")


def cell(text: str, *, label: bool = False) -> Paragraph:
    return Paragraph(text, cell_label_style if label else cell_value_style)


def build_pdf(claim: dict, insurer: str) -> None:
    out_path = OUTPUT_DIR / f"{claim['id']}.pdf"
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        topMargin=1.8 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
    )

    rows = [
        ["Claimant Name", claim["claimant_name"], "Claimant ID", claim["claimant_id"]],
        ["Policy Number", claim["policy_no"], "Policy Start Date", format_date(claim["policy_start_date"])],
        ["Date of Loss / Admission", format_date(claim["date_of_loss"]), "Date Filed", format_date(claim["date_filed"])],
        ["Claim Category", claim["category"], "Amount Claimed (INR)", f"Rs. {claim['amount_inr']:,}"],
        ["Diagnosis", claim["diagnosis"], "Hospital / Provider", claim["hospital"]],
        ["Treatment / Procedure", claim["treatment"], "", ""],
    ]
    table_data = [
        [cell(r[0], label=True), cell(r[1]), cell(r[2], label=True) if r[2] else "", cell(r[3]) if r[3] else ""]
        for r in rows
    ]

    elements = [
        Paragraph(insurer, insurer_style),
        Paragraph("HEALTH INSURANCE CLAIM FORM", title_style),
        Paragraph(
            "SYNTHETIC DATA — FOR PORTFOLIO / DEMO PURPOSES ONLY. NOT A REAL CLAIM, CLAIMANT, OR INSURER.",
            disclaimer_style,
        ),
        Table(
            table_data,
            colWidths=[4 * cm, 5 * cm, 4 * cm, 5 * cm],
            style=TABLE_STYLE,
        ),
        Paragraph("Description of Incident / Narrative", section_style),
        Paragraph(claim["narrative"], body_style),
        Spacer(1, 24),
        Paragraph(
            "This is a synthetic document generated for the Claims Copilot: Fraud Triage Agent portfolio project. "
            "No real individuals, policies, or medical records are represented.",
            footer_style,
        ),
    ]

    doc.build(elements)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    insurer = source["insurer"]

    print(f"Generating {len(source['claims'])} claim PDFs into {OUTPUT_DIR.relative_to(ROOT)}/")
    for claim in source["claims"]:
        build_pdf(claim, insurer)
    print("Done.")


if __name__ == "__main__":
    main()
