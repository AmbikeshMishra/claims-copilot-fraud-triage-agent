"""Pipeline-specific exceptions surfaced as friendly errors in app.py, rather
than raw stack traces, for the Day 5 edge cases (brief §7): unparseable PDFs
and failed/malformed extraction.
"""


class PDFTextExtractionError(Exception):
    """Raised when a PDF can't be opened or contains no extractable text."""


class ExtractionError(Exception):
    """Raised when the LLM extraction step fails or returns invalid fields."""
