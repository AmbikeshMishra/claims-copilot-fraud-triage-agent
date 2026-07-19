"""Score & triage node (brief §3c): transparent weighted rules — no black-box
ML in v1; explainability is the point. Weights below were tuned against the
9 ground-truth samples in data/claims_source.json (see expected_bucket /
expected_signals) and can be replaced with a trained model later (README note).
"""

from pydantic import BaseModel

from .signals import Signal

SIGNAL_WEIGHTS: dict[str, int] = {
    "early_claim": 20,
    "high_amount_vs_category_median": 25,
    "repeat_claimant": 20,
    "narrative_inconsistency": 30,
    "treatment_diagnosis_mismatch": 35,
}

AUTO_APPROVE_MAX = 19
STANDARD_REVIEW_MAX = 59

# Bucket transitions sit at the midpoint between the last score of one bucket
# and the first of the next; a score within this many points of a transition
# is flagged as a borderline verdict (brief §7 Day 5: "ambiguous verdicts near
# bucket boundaries") since a small change in the facts could flip the bucket.
BUCKET_TRANSITIONS = (AUTO_APPROVE_MAX + 0.5, STANDARD_REVIEW_MAX + 0.5)
BORDERLINE_MARGIN = 5

BORDERLINE_NOTES = {
    AUTO_APPROVE_MAX + 0.5: (
        "This score sits close to the AUTO-APPROVE / STANDARD REVIEW boundary — a small change in "
        "the facts could shift the bucket either way; apply extra adjuster judgment before finalizing."
    ),
    STANDARD_REVIEW_MAX + 0.5: (
        "This score sits close to the STANDARD REVIEW / INVESTIGATE boundary — a small change in the "
        "facts could shift the bucket either way; apply extra adjuster judgment before finalizing."
    ),
}


class TriageResult(BaseModel):
    score: int
    bucket: str
    borderline: bool = False
    borderline_note: str | None = None


def score_signals(signals: list[Signal]) -> TriageResult:
    raw_score = sum(SIGNAL_WEIGHTS.get(s.name, 0) for s in signals if s.fired)
    score = min(raw_score, 100)
    if score <= AUTO_APPROVE_MAX:
        bucket = "AUTO-APPROVE"
    elif score <= STANDARD_REVIEW_MAX:
        bucket = "STANDARD REVIEW"
    else:
        bucket = "INVESTIGATE"

    nearest_transition = min(BUCKET_TRANSITIONS, key=lambda t: abs(score - t))
    borderline = abs(score - nearest_transition) <= BORDERLINE_MARGIN
    borderline_note = BORDERLINE_NOTES[nearest_transition] if borderline else None

    return TriageResult(score=score, bucket=bucket, borderline=borderline, borderline_note=borderline_note)
