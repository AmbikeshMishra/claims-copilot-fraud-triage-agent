"""Generate synthetic edge-case PDF fixtures for Day 5 (brief §7: unparseable
PDF handling). Kept in data/edge_case_pdfs/, separate from data/claims_pdfs/,
so they never touch the 9-sample ground-truth flow validate_extraction.py /
validate_signals.py rely on.

Usage:
    python scripts/generate_edge_case_pdfs.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "edge_case_pdfs"


def build_no_text_layer_pdf(out_path: Path) -> None:
    """A valid PDF page drawn only from vector shapes (no drawString/Paragraph
    calls), so it carries zero extractable text — simulating a scanned claim
    form with no OCR text layer.
    """
    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    c.setStrokeColorRGB(0.6, 0.6, 0.6)
    c.rect(2 * 72, height - 3 * 72, width - 4 * 72, 72)
    for i in range(6):
        y = height - 5 * 72 - i * 0.4 * 72
        c.line(1.5 * 72, y, width - 1.5 * 72, y)
    c.showPage()
    c.save()


def build_corrupt_pdf(out_path: Path) -> None:
    """A few bytes that aren't a valid PDF at all, so fitz.open() itself raises."""
    out_path.write_bytes(b"this is not a real pdf file - synthetic edge-case fixture\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_no_text_layer_pdf(OUTPUT_DIR / "no_text_layer.pdf")
    print(f"  wrote {(OUTPUT_DIR / 'no_text_layer.pdf').relative_to(ROOT)}")
    build_corrupt_pdf(OUTPUT_DIR / "corrupt.pdf")
    print(f"  wrote {(OUTPUT_DIR / 'corrupt.pdf').relative_to(ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
