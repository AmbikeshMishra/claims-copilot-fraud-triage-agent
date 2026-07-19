"""LangGraph pipeline (brief §3): extract -> signal check -> score & triage -> report.

All four nodes are wired up as of Day 4.
"""

from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .extraction import extract_claim_fields
from .pdf_text import extract_text
from .report import build_report_markdown, build_report_pdf
from .schema import ExtractedClaim
from .scoring import score_signals
from .signals import Signal, run_signal_checks


class PipelineState(TypedDict, total=False):
    pdf_path: str | bytes
    api_key: str
    claim_id: str
    raw_text: str
    extracted: ExtractedClaim
    signals: list[Signal]
    score: int
    bucket: str
    borderline: bool
    borderline_note: str | None
    report_markdown: str
    report_pdf: bytes


def extract_node(state: PipelineState) -> PipelineState:
    source = state["pdf_path"]
    raw_text = extract_text(source if isinstance(source, (bytes, bytearray)) else Path(source))
    extracted = extract_claim_fields(raw_text, state["api_key"])
    return {"raw_text": raw_text, "extracted": extracted}


def signal_check_node(state: PipelineState) -> PipelineState:
    signals = run_signal_checks(state["extracted"], state["api_key"])
    return {"signals": signals}


def score_triage_node(state: PipelineState) -> PipelineState:
    result = score_signals(state["signals"])
    return {
        "score": result.score,
        "bucket": result.bucket,
        "borderline": result.borderline,
        "borderline_note": result.borderline_note,
    }


def report_node(state: PipelineState) -> PipelineState:
    pdf_path = state["pdf_path"]
    claim_id = state.get("claim_id") or (Path(pdf_path).stem if isinstance(pdf_path, str) else "claim")
    claim, signals, score, bucket = state["extracted"], state["signals"], state["score"], state["bucket"]
    borderline, borderline_note = state.get("borderline", False), state.get("borderline_note")
    markdown = build_report_markdown(claim_id, claim, signals, score, bucket, borderline, borderline_note)
    pdf_bytes = build_report_pdf(claim_id, claim, signals, score, bucket, borderline, borderline_note)
    return {"report_markdown": markdown, "report_pdf": pdf_bytes}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("extract", extract_node)
    graph.add_node("signal_check", signal_check_node)
    graph.add_node("score_triage", score_triage_node)
    graph.add_node("report", report_node)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "signal_check")
    graph.add_edge("signal_check", "score_triage")
    graph.add_edge("score_triage", "report")
    graph.add_edge("report", END)
    return graph.compile()
