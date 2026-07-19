"""Raw text extraction from claim PDFs via PyMuPDF."""

from pathlib import Path

import fitz  # PyMuPDF

from .errors import PDFTextExtractionError


def extract_text(source: Path | str | bytes) -> str:
    try:
        if isinstance(source, (bytes, bytearray)):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            doc = fitz.open(source)
    except Exception as exc:
        raise PDFTextExtractionError(
            "Could not open this PDF — the file may be corrupted or not a valid PDF."
        ) from exc

    with doc:
        text = "\n".join(page.get_text() for page in doc)

    if not text.strip():
        raise PDFTextExtractionError(
            "No extractable text found in this PDF. It may be a scanned image with no text "
            "layer — this app requires text-based PDFs."
        )
    return text


def render_pages_as_png(source: Path | str | bytes, zoom: float = 1.6) -> list[bytes]:
    """Render each page to PNG bytes for the UI preview.

    Streamlit Community Cloud's hosting iframe blocks nested data: URI
    iframes via CSP, so a raw-PDF <iframe> preview (which works locally)
    renders as a blocked-content icon there. Rendering to images sidesteps
    that entirely and needs no browser PDF viewer.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            doc = fitz.open(source)
    except Exception as exc:
        raise PDFTextExtractionError(
            "Could not open this PDF — the file may be corrupted or not a valid PDF."
        ) from exc

    matrix = fitz.Matrix(zoom, zoom)
    with doc:
        return [page.get_pixmap(matrix=matrix).tobytes("png") for page in doc]
