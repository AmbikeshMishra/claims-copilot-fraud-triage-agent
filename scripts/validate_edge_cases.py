"""Validate the Day 5 edge-case handling (brief §7): unparseable PDF, missing/
malformed fields, and ambiguous verdicts near bucket boundaries.

Usage:
    python scripts/validate_edge_cases.py
    (does not call the LLM — no OPENAI_API_KEY required)
"""

import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.errors import PDFTextExtractionError  # noqa: E402
from src.pdf_text import extract_text  # noqa: E402
from src.schema import ExtractedClaim  # noqa: E402
from src.scoring import score_signals  # noqa: E402
from src.signals import Signal, check_early_claim, check_repeat_claimant  # noqa: E402

EDGE_CASE_DIR = ROOT / "data" / "edge_case_pdfs"

VALID_CLAIM_KWARGS = dict(
    claimant_name="Test Claimant",
    policy_no="POL-TEST-0001",
    date_of_loss="2026-01-01",
    category="Illness / Hospitalization (Surgery)",
    amount_inr=10000,
    narrative="Test narrative long enough to pass extraction guards.",
)

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}{f' — {detail}' if detail and not condition else ''}")
    if not condition:
        failures.append(label)


def main() -> None:
    if not (EDGE_CASE_DIR / "no_text_layer.pdf").exists() or not (EDGE_CASE_DIR / "corrupt.pdf").exists():
        print("Edge-case fixtures not found. Run scripts/generate_edge_case_pdfs.py first.")
        sys.exit(1)

    # 1. No-text-layer PDF -> PDFTextExtractionError
    try:
        extract_text(EDGE_CASE_DIR / "no_text_layer.pdf")
        check("no_text_layer.pdf raises PDFTextExtractionError", False)
    except PDFTextExtractionError:
        check("no_text_layer.pdf raises PDFTextExtractionError", True)
    except Exception as exc:  # noqa: BLE001
        check("no_text_layer.pdf raises PDFTextExtractionError", False, f"raised {type(exc).__name__} instead")

    # 2. Corrupt PDF -> PDFTextExtractionError
    try:
        extract_text(EDGE_CASE_DIR / "corrupt.pdf")
        check("corrupt.pdf raises PDFTextExtractionError", False)
    except PDFTextExtractionError:
        check("corrupt.pdf raises PDFTextExtractionError", True)
    except Exception as exc:  # noqa: BLE001
        check("corrupt.pdf raises PDFTextExtractionError", False, f"raised {type(exc).__name__} instead")

    # 3. Malformed date rejected by schema validator
    try:
        ExtractedClaim(**{**VALID_CLAIM_KWARGS, "date_of_loss": "not-a-date"})
        check("malformed date_of_loss rejected by ExtractedClaim", False)
    except ValidationError:
        check("malformed date_of_loss rejected by ExtractedClaim", True)

    # 4. Borderline scoring
    def signals_with(*names: str) -> list[Signal]:
        return [Signal(name=n, fired=True, explanation="") for n in names]

    score_20 = score_signals(signals_with("early_claim"))
    check(
        "score=20 (AUTO/STANDARD boundary) flagged borderline",
        score_20.score == 20 and score_20.borderline is True,
        f"score={score_20.score} borderline={score_20.borderline}",
    )

    score_0 = score_signals([])
    check(
        "score=0 not flagged borderline",
        score_0.score == 0 and score_0.borderline is False,
        f"score={score_0.score} borderline={score_0.borderline}",
    )

    score_45 = score_signals(signals_with("high_amount_vs_category_median", "repeat_claimant"))
    check(
        "score=45 (mid STANDARD REVIEW) not flagged borderline",
        score_45.score == 45 and score_45.borderline is False,
        f"score={score_45.score} borderline={score_45.borderline}",
    )

    # 5. Defensive date parsing in signals — malformed date should skip, not raise.
    # Direct attribute assignment bypasses the schema validator (validate_assignment
    # is off by default) — simulating a bad date that slipped past extraction-time
    # validation, e.g. via a hand-built/legacy record.
    claim_bad_dates = ExtractedClaim(**{**VALID_CLAIM_KWARGS, "policy_start_date": "2025-01-01"})
    claim_bad_dates.policy_start_date = "not-a-date"
    try:
        signal = check_early_claim(claim_bad_dates)
        check("check_early_claim tolerates malformed date", signal.fired is False)
    except Exception as exc:  # noqa: BLE001
        check("check_early_claim tolerates malformed date", False, f"raised {type(exc).__name__}: {exc}")

    import pandas as pd

    bad_history = pd.DataFrame({"claimant_id": ["CUST-X"], "claim_date": ["not-a-date"]})
    claim_repeat = ExtractedClaim(**{**VALID_CLAIM_KWARGS, "claimant_id": "CUST-X"})
    try:
        signal = check_repeat_claimant(claim_repeat, bad_history)
        check("check_repeat_claimant tolerates malformed history date", signal.fired is False)
    except Exception as exc:  # noqa: BLE001
        check("check_repeat_claimant tolerates malformed history date", False, f"raised {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        sys.exit(1)
    print("All edge-case checks passed.")


if __name__ == "__main__":
    main()
