"""Session-scoped cost guard for LLM calls triggered from the Streamlit UI (brief §5)."""

import time

import streamlit as st

MIN_INTERVAL_SECONDS = 3
MAX_CALLS_PER_SESSION = 20


def throttle_block_reason() -> str | None:
    """Returns a user-facing reason to block the next LLM call, or None if it's allowed."""
    count = st.session_state.get("llm_call_count", 0)
    last_call = st.session_state.get("last_llm_call_time", 0.0)
    now = time.time()

    if count >= MAX_CALLS_PER_SESSION:
        return f"Session limit of {MAX_CALLS_PER_SESSION} LLM calls reached. Refresh the app to reset."
    elapsed = now - last_call
    if elapsed < MIN_INTERVAL_SECONDS:
        return f"Cost guard: please wait {MIN_INTERVAL_SECONDS - elapsed:.1f}s before the next LLM call."
    return None


def record_call() -> None:
    st.session_state["llm_call_count"] = st.session_state.get("llm_call_count", 0) + 1
    st.session_state["last_llm_call_time"] = time.time()
