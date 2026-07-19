"""Run the Extract node against every sample claim PDF and diff against ground truth.

Usage:
    python scripts/validate_extraction.py
    (reads OPENAI_API_KEY from .env / environment)
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph import build_graph  # noqa: E402

load_dotenv()

SOURCE_PATH = ROOT / "data" / "claims_source.json"
PDF_DIR = ROOT / "data" / "claims_pdfs"

# Ground-truth field name -> extracted field name, where they differ.
FIELD_MAP = {
    "claimant_name": "claimant_name",
    "claimant_id": "claimant_id",
    "policy_no": "policy_no",
    "policy_start_date": "policy_start_date",
    "date_of_loss": "date_of_loss",
    "date_filed": "date_filed",
    "category": "category",
    "amount_inr": "amount_inr",
    "diagnosis": "diagnosis",
    "treatment": "treatment",
    "hospital": "hospital",
}


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set (put it in .env or the environment). Aborting.")
        sys.exit(1)

    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    graph = build_graph()

    total_fields = 0
    mismatched_fields = 0
    claims_with_mismatches = 0

    for claim in source["claims"]:
        pdf_path = PDF_DIR / f"{claim['id']}.pdf"
        if not pdf_path.exists():
            print(f"[{claim['id']}] SKIP - PDF not found at {pdf_path}")
            continue

        result = graph.invoke({"pdf_path": str(pdf_path), "api_key": api_key})
        extracted = result["extracted"].model_dump()

        mismatches = []
        for truth_field, extracted_field in FIELD_MAP.items():
            expected = claim.get(truth_field)
            actual = extracted.get(extracted_field)
            total_fields += 1
            if str(expected) != str(actual):
                mismatches.append((truth_field, expected, actual))
                mismatched_fields += 1

        status = "OK" if not mismatches else f"{len(mismatches)} MISMATCH(ES)"
        print(f"[{claim['id']}] {status}")
        for field, expected, actual in mismatches:
            print(f"    {field}: expected={expected!r} got={actual!r}")
        if mismatches:
            claims_with_mismatches += 1

    print()
    print(f"Fields checked: {total_fields}, mismatched: {mismatched_fields}")
    print(f"Claims with at least one mismatch: {claims_with_mismatches} / {len(source['claims'])}")


if __name__ == "__main__":
    main()
