"""Run the signal-check + score-triage nodes against every sample claim's
ground-truth fields and diff against expected_signals / expected_bucket.

Uses ground-truth fields directly (not the Extract node) so this validates
signal/scoring logic in isolation from extraction accuracy, which
validate_extraction.py already covers.

Usage:
    python scripts/validate_signals.py
    (reads OPENAI_API_KEY from .env / environment)
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schema import ExtractedClaim  # noqa: E402
from src.scoring import score_signals  # noqa: E402
from src.signals import run_signal_checks  # noqa: E402

load_dotenv()

SOURCE_PATH = ROOT / "data" / "claims_source.json"

CLAIM_FIELDS = [
    "claimant_name",
    "claimant_id",
    "policy_no",
    "policy_start_date",
    "date_of_loss",
    "date_filed",
    "category",
    "amount_inr",
    "diagnosis",
    "treatment",
    "hospital",
    "narrative",
]


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set (put it in .env or the environment). Aborting.")
        sys.exit(1)

    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))

    claims_ok = 0
    total = len(source["claims"])

    for claim_data in source["claims"]:
        extracted = ExtractedClaim(**{field: claim_data[field] for field in CLAIM_FIELDS})
        signals = run_signal_checks(extracted, api_key)
        result = score_signals(signals)

        fired_names = {s.name for s in signals if s.fired}
        expected_names = set(claim_data["expected_signals"])
        expected_bucket = claim_data["expected_bucket"]

        signals_ok = fired_names == expected_names
        bucket_ok = result.bucket == expected_bucket
        ok = signals_ok and bucket_ok

        status = "OK" if ok else "MISMATCH"
        print(f"[{claim_data['id']}] {status}  score={result.score} bucket={result.bucket} (expected {expected_bucket})")
        if not signals_ok:
            missing = expected_names - fired_names
            extra = fired_names - expected_names
            if missing:
                print(f"    missing signals: {sorted(missing)}")
            if extra:
                print(f"    unexpected signals: {sorted(extra)}")
        if ok:
            claims_ok += 1

    print()
    print(f"Claims matching ground truth: {claims_ok} / {total}")


if __name__ == "__main__":
    main()
