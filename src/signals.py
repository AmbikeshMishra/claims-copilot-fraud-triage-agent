"""Signal check node (brief §3b): explainable fraud signals against extracted
fields, category medians, and a synthetic claims-history table.

Three signals are plain rule-based comparisons (dates, amounts, history
lookups). Two — narrative_inconsistency and treatment_diagnosis_mismatch —
require semantic judgment a keyword rule can't reliably deliver, so they run
as a single LLM call that returns both flags with evidence quotes, keeping
this node to one LLM call for the throttle/cost guard.
"""

from datetime import date
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .schema import ExtractedClaim

ROOT = Path(__file__).resolve().parent.parent
CLAIMS_HISTORY_PATH = ROOT / "data" / "claims_history.csv"
CATEGORY_MEDIANS_PATH = ROOT / "data" / "category_medians.csv"

EARLY_CLAIM_DAYS = 30
HIGH_AMOUNT_MULTIPLIER = 2.0
REPEAT_CLAIMANT_LOOKBACK_DAYS = 365


class Signal(BaseModel):
    name: str
    fired: bool
    explanation: str
    evidence: str | None = None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def check_early_claim(claim: ExtractedClaim) -> Signal:
    if not claim.policy_start_date:
        return Signal(
            name="early_claim", fired=False, explanation="Policy start date not available on this claim; check skipped."
        )
    loss_date, start_date = _parse_date(claim.date_of_loss), _parse_date(claim.policy_start_date)
    if loss_date is None or start_date is None:
        return Signal(
            name="early_claim", fired=False, explanation="Loss or policy start date could not be parsed; check skipped."
        )
    days = (loss_date - start_date).days
    fired = 0 <= days < EARLY_CLAIM_DAYS
    if fired:
        explanation = (
            f"Loss occurred just {days} day(s) after the policy started (start {claim.policy_start_date}, "
            f"loss {claim.date_of_loss}) — claims this early in a policy carry higher non-disclosure/fraud risk."
        )
    else:
        explanation = (
            f"Loss occurred {days} day(s) after the policy started — outside the {EARLY_CLAIM_DAYS}-day "
            "early-claim window."
        )
    return Signal(name="early_claim", fired=fired, explanation=explanation)


def check_high_amount(claim: ExtractedClaim, medians: dict[str, float]) -> Signal:
    median = medians.get(claim.category)
    if not median:
        return Signal(
            name="high_amount_vs_category_median",
            fired=False,
            explanation=f"No category median on file for '{claim.category}'; check skipped.",
        )
    ratio = claim.amount_inr / median
    fired = ratio > HIGH_AMOUNT_MULTIPLIER
    if fired:
        explanation = (
            f"Claimed amount Rs. {claim.amount_inr:,} is {ratio:.1f}x the '{claim.category}' category median "
            f"of Rs. {median:,.0f} — unusually high for this type of claim."
        )
    else:
        explanation = (
            f"Claimed amount Rs. {claim.amount_inr:,} is {ratio:.1f}x the '{claim.category}' category median "
            f"of Rs. {median:,.0f} — within a normal range."
        )
    return Signal(name="high_amount_vs_category_median", fired=fired, explanation=explanation)


def check_repeat_claimant(claim: ExtractedClaim, history: pd.DataFrame) -> Signal:
    if not claim.claimant_id:
        return Signal(name="repeat_claimant", fired=False, explanation="Claimant ID not available; check skipped.")
    loss_date = _parse_date(claim.date_of_loss)
    if loss_date is None:
        return Signal(name="repeat_claimant", fired=False, explanation="Date of loss could not be parsed; check skipped.")
    prior = history[history["claimant_id"] == claim.claimant_id]
    count = 0
    for claim_date_str in prior["claim_date"]:
        prior_date = _parse_date(str(claim_date_str))
        if prior_date is None:
            continue
        if prior_date <= loss_date and (loss_date - prior_date).days <= REPEAT_CLAIMANT_LOOKBACK_DAYS:
            count += 1
    fired = count >= 1
    if fired:
        explanation = (
            f"Claimant has {count} prior claim(s) on file in the past {REPEAT_CLAIMANT_LOOKBACK_DAYS} days — "
            "repeat claimants warrant a closer look at the pattern, not automatic denial."
        )
    else:
        explanation = "No prior claims found for this claimant in the past 12 months."
    return Signal(name="repeat_claimant", fired=fired, explanation=explanation)


class NarrativeAssessment(BaseModel):
    narrative_inconsistency: bool = Field(
        description="True if the incident narrative is internally inconsistent, vague on key facts "
        "(location, timeline, cause), or the story doesn't hold together on its own terms."
    )
    narrative_evidence: str | None = Field(
        default=None, description="Verbatim quote from the narrative supporting the finding, or null if not fired."
    )
    narrative_explanation: str = Field(description="One-sentence explanation in the voice of an experienced adjuster.")
    treatment_diagnosis_mismatch: bool = Field(
        description="True only if the treatment is a fundamentally different kind of intervention than the "
        "diagnosis calls for — e.g. a diagnosis consistent with a minor/viral illness but the treatment is "
        "major cardiac or other organ surgery unrelated to that diagnosis. Judge this purely from the medical "
        "relationship between diagnosis and treatment — IGNORE claim amount/cost entirely (a separate signal "
        "scores that). Hospitalization, observation, casting, reduction, or surgery that directly treats the "
        "stated injury/condition (e.g. hospitalization for a fall injury, surgery for the diagnosed condition) "
        "is standard care and is NOT a mismatch, no matter how routinely or vaguely it's described."
    )
    treatment_evidence: str | None = Field(
        default=None,
        description="Verbatim quote (diagnosis and/or treatment/narrative) supporting the finding, or null if not fired.",
    )
    treatment_explanation: str = Field(description="One-sentence explanation in the voice of an experienced adjuster.")


NARRATIVE_SYSTEM_PROMPT = (
    "You are an experienced health insurance claims adjuster screening a claim for red flags. Using only the "
    "diagnosis, treatment, and narrative given, assess two independent things: "
    "(1) whether the incident narrative is internally consistent and specific, or vague/contradictory in a way "
    "that would concern an adjuster; "
    "(2) whether the stated treatment represents a clear, unexplained medical escalation in severity or scope "
    "beyond what the stated diagnosis would call for — judge this purely on the diagnosis/treatment relationship, "
    "ignoring cost and amount entirely (a separate check already handles cost). "
    "Quote evidence verbatim from the text provided. Be conservative — flag genuine concerns only, not stylistic "
    "quirks, brevity, cost, or routine medical detail. Write explanations the way an adjuster would in a case "
    "note, in plain English, one sentence each."
)


def check_narrative_and_treatment(claim: ExtractedClaim, api_key: str, model: str = "gpt-4o-mini") -> tuple[Signal, Signal]:
    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    structured_llm = llm.with_structured_output(NarrativeAssessment)
    human = f"Diagnosis: {claim.diagnosis}\nTreatment: {claim.treatment}\nNarrative: {claim.narrative}"
    result = structured_llm.invoke([("system", NARRATIVE_SYSTEM_PROMPT), ("human", human)])
    assert isinstance(result, NarrativeAssessment)
    narrative_signal = Signal(
        name="narrative_inconsistency",
        fired=result.narrative_inconsistency,
        explanation=result.narrative_explanation,
        evidence=result.narrative_evidence,
    )
    treatment_signal = Signal(
        name="treatment_diagnosis_mismatch",
        fired=result.treatment_diagnosis_mismatch,
        explanation=result.treatment_explanation,
        evidence=result.treatment_evidence,
    )
    return narrative_signal, treatment_signal


def load_category_medians() -> dict[str, float]:
    df = pd.read_csv(CATEGORY_MEDIANS_PATH)
    return dict(zip(df["category"], df["median_amount_inr"]))


def load_claims_history() -> pd.DataFrame:
    return pd.read_csv(CLAIMS_HISTORY_PATH)


def run_signal_checks(claim: ExtractedClaim, api_key: str) -> list[Signal]:
    medians = load_category_medians()
    history = load_claims_history()
    narrative_signal, treatment_signal = check_narrative_and_treatment(claim, api_key)
    return [
        check_early_claim(claim),
        check_high_amount(claim, medians),
        check_repeat_claimant(claim, history),
        narrative_signal,
        treatment_signal,
    ]
