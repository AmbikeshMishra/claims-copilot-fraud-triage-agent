"""Structured field schema for the Extract node.

Superset of the brief's minimum (claimant, policy no., date of loss, claim
type, amount, narrative) — also captures policy_start_date, date_filed,
diagnosis, treatment, and claimant_id, since the Day 3 signal engine needs
them (early_claim, treatment_diagnosis_mismatch, repeat_claimant) and they're
already present on every sample claim form.
"""

from datetime import date

from pydantic import BaseModel, Field, field_validator

DATE_FIELDS = ("policy_start_date", "date_of_loss", "date_filed")


class ExtractedClaim(BaseModel):
    claimant_name: str = Field(description="Full name of the insured/claimant")
    claimant_id: str | None = Field(default=None, description="Claimant/customer ID, e.g. CUST-20001")
    policy_no: str = Field(description="Policy number")
    policy_start_date: str | None = Field(
        default=None, description="Policy start date, normalized to ISO format YYYY-MM-DD"
    )
    date_of_loss: str = Field(description="Date of loss / admission, normalized to ISO format YYYY-MM-DD")
    date_filed: str | None = Field(
        default=None, description="Date the claim was filed, normalized to ISO format YYYY-MM-DD"
    )
    category: str = Field(description="Claim category / type exactly as stated on the form")
    amount_inr: int = Field(description="Amount claimed in INR as a plain integer (no currency symbol, no commas)")
    diagnosis: str | None = Field(default=None, description="Diagnosis stated on the claim form")
    treatment: str | None = Field(default=None, description="Treatment or procedure performed")
    hospital: str | None = Field(default=None, description="Hospital or provider name")
    narrative: str = Field(description="The incident/narrative section, verbatim from the document")

    @field_validator(*DATE_FIELDS)
    @classmethod
    def _validate_iso_date(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"'{value}' is not a valid ISO date (YYYY-MM-DD)") from exc
        return value
