"""LLM field structuring for the Extract node (brief §3a)."""

from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from .errors import ExtractionError
from .schema import ExtractedClaim

EXTRACTION_SYSTEM_PROMPT = (
    "You are an insurance claims data-entry assistant. Read the raw text of a health "
    "insurance claim form and extract the structured fields exactly as they appear on "
    "the document. Do not infer, guess, or invent values that are not present in the "
    "text — leave optional fields null if absent. Normalize all dates to ISO format "
    "YYYY-MM-DD. The narrative field must be the verbatim incident narrative text, not "
    "a summary or paraphrase."
)

MIN_RAW_TEXT_CHARS = 20


def extract_claim_fields(raw_text: str, api_key: str, model: str = "gpt-4o-mini") -> ExtractedClaim:
    if len(raw_text.strip()) < MIN_RAW_TEXT_CHARS:
        raise ExtractionError(
            "This document doesn't contain enough text to extract claim fields from."
        )

    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    structured_llm = llm.with_structured_output(ExtractedClaim)
    try:
        result = structured_llm.invoke(
            [
                ("system", EXTRACTION_SYSTEM_PROMPT),
                ("human", raw_text),
            ]
        )
    except ValidationError as exc:
        fields = ", ".join(sorted({str(err["loc"][0]) for err in exc.errors() if err["loc"]}))
        raise ExtractionError(
            f"The model couldn't extract valid claim fields from this document"
            f"{f' (problem fields: {fields})' if fields else ''}."
        ) from exc
    except Exception as exc:
        raise ExtractionError(f"Claim field extraction failed: {exc}") from exc

    assert isinstance(result, ExtractedClaim)
    return result
