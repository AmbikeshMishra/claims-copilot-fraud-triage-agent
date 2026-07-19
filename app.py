"""Claims Copilot: Fraud Triage Agent — Streamlit shell.

Day 1: sample-claim picker + document preview.
Day 2: Extract node (PDF -> structured fields via LLM) wired in below.
Day 3: Signal check + score/triage nodes wired in below.
Day 4: Report node wired in below (downloadable markdown/PDF report).
Day 5: File upload + friendly handling of unparseable PDFs, failed
extraction, and borderline (near-boundary) verdicts.
"""

import json
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.errors import ExtractionError, PDFTextExtractionError
from src.graph import build_graph
from src.pdf_text import render_pages_as_png
from src.throttle import record_call, throttle_block_reason

load_dotenv()

ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "data" / "claims_source.json"
PDF_DIR = ROOT / "data" / "claims_pdfs"

st.set_page_config(page_title="Claims Copilot: Fraud Triage Agent", layout="wide")


@st.cache_data
def load_claims() -> list[dict]:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    return source["claims"]


@st.cache_data(show_spinner=False)
def _render_pdf_pages(pdf_bytes: bytes) -> list[bytes]:
    return render_pages_as_png(pdf_bytes)


def render_pdf_preview(pdf_bytes: bytes) -> None:
    try:
        pages = _render_pdf_pages(pdf_bytes)
    except PDFTextExtractionError as exc:
        st.warning(str(exc))
        return
    for page_png in pages:
        st.image(page_png, use_container_width=True)


def sanitize_claim_id(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "uploaded_claim"


st.title("Claims Copilot: Fraud Triage Agent")
st.caption(
    "Decision-support demo, not an underwriting or denial system — a human adjuster makes the final call. "
    "All claims shown are synthetic; no real claimants, policies, or medical records."
)

claims = load_claims()
labels = [f"{c['id']} — {c['claimant_name']} ({c['category']}, {c['variant']})" for c in claims]
selected_label = st.sidebar.selectbox("Sample claim", labels)
selected_claim = claims[labels.index(selected_label)]

uploaded_file = st.sidebar.file_uploader(
    "Or upload your own claim PDF",
    type=["pdf"],
    help="Uploaded files are processed only for this session and never stored.",
)

st.sidebar.markdown("---")
api_key = st.sidebar.text_input(
    "OpenAI API key",
    value=os.environ.get("OPENAI_API_KEY", ""),
    type="password",
    help="Used only for this session, never stored. Get one at platform.openai.com.",
)
st.sidebar.caption("Run triage on a claim to generate a downloadable one-page report.")

if uploaded_file is not None:
    active_id = sanitize_claim_id(uploaded_file.name)
    active_pdf_bytes = uploaded_file.getvalue()
    active_pdf_source: str | bytes = active_pdf_bytes
    pdf_available = True
else:
    active_id = selected_claim["id"]
    pdf_path = PDF_DIR / f"{selected_claim['id']}.pdf"
    pdf_available = pdf_path.exists()
    active_pdf_bytes = pdf_path.read_bytes() if pdf_available else None
    active_pdf_source = str(pdf_path)

col_preview, col_meta = st.columns([2, 1])

with col_preview:
    st.subheader("Claim document")
    if pdf_available:
        render_pdf_preview(active_pdf_bytes)
    elif uploaded_file is None:
        st.warning(
            f"PDF not found at {pdf_path.relative_to(ROOT)}. "
            "Run `python scripts/generate_claim_pdfs.py` to generate the sample claim PDFs."
        )

with col_meta:
    st.subheader("Claim summary")
    if uploaded_file is not None:
        st.info("Uploaded document — run triage to extract its fields below.")
    else:
        st.write(f"**Claimant:** {selected_claim['claimant_name']}")
        st.write(f"**Category:** {selected_claim['category']}")
        st.write(f"**Amount claimed:** Rs. {selected_claim['amount_inr']:,}")
        st.write(f"**Date of loss:** {selected_claim['date_of_loss']}")

    st.markdown("---")
    st.subheader("Extracted fields")

    triage_by_claim = st.session_state.setdefault("triage_by_claim", {})
    run_triage = st.button("Run triage", disabled=not pdf_available)

    if run_triage:
        if not api_key:
            st.error("Enter an OpenAI API key in the sidebar first.")
        else:
            block_reason = throttle_block_reason()
            if block_reason:
                st.warning(block_reason)
            else:
                with st.spinner("Running extract -> signal check -> score & triage -> report..."):
                    try:
                        graph = build_graph()
                        result = graph.invoke(
                            {"pdf_path": active_pdf_source, "api_key": api_key, "claim_id": active_id}
                        )
                        record_call()
                        triage_by_claim[active_id] = result
                    except PDFTextExtractionError as exc:
                        st.error(f"Couldn't read this PDF: {exc}")
                    except ExtractionError as exc:
                        st.error(f"Couldn't extract claim fields: {exc}")
                    except Exception as exc:  # noqa: BLE001 - surface any other LLM/parsing failure to the user
                        st.error(f"Triage run failed: {exc}")

    result = triage_by_claim.get(active_id)
    if result is None:
        st.info("Click **Run triage** to run this claim through the extract -> signal check -> score pipeline.")
    else:
        fields = result["extracted"].model_dump()
        st.table({"field": list(fields.keys()), "value": [str(v) for v in fields.values()]})

        st.markdown("---")
        st.subheader("Signal checklist")
        for signal in result["signals"]:
            icon = "\U0001F6A9" if signal.fired else "✅"
            st.markdown(f"{icon} **{signal.name.replace('_', ' ')}** — {signal.explanation}")
            if signal.fired and signal.evidence:
                st.caption(f'Evidence: "{signal.evidence}"')

        st.markdown("---")
        bucket = result["bucket"]
        banner = {"AUTO-APPROVE": st.success, "STANDARD REVIEW": st.warning, "INVESTIGATE": st.error}[bucket]
        banner(f"**{bucket}** — risk score {result['score']}/100")
        if result.get("borderline"):
            st.caption(f"⚠️ {result['borderline_note']}")

        st.markdown("---")
        st.subheader("Triage report")
        dl_col_md, dl_col_pdf = st.columns(2)
        with dl_col_md:
            st.download_button(
                "Download report (Markdown)",
                data=result["report_markdown"],
                file_name=f"{active_id}_triage_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl_col_pdf:
            st.download_button(
                "Download report (PDF)",
                data=result["report_pdf"],
                file_name=f"{active_id}_triage_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with st.expander("Preview report"):
            st.markdown(result["report_markdown"])
