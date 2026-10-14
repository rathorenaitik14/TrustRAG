"""
Manages Streamlit session state initialization and access helpers for TRUSTRAG.
"""
import streamlit as st

DEFAULTS = {
    # Document state
    "uploaded_file_name": None,
    "document_pages": 0,
    "document_chunks": 0,
    "embedding_dim": 0,
    "retriever": None,
    "reranker": None,
    "ev_retriever": None,
    "verifier": None,
    "corrector": None,
    "report_generator": None,
    # Query state
    "question": "",
    "pipeline_ran": False,
    # Pipeline outputs
    "retrieved_chunks": None,
    "reranked_chunks": None,
    "context": None,
    "original_answer": None,
    "claims": None,
    "evidence_results": None,
    "verification_results": None,
    "correction_changes": None,
    "corrected_answer": None,
    "reliability_report": None,
    # Stage latencies (seconds)
    "latency": {},
    # Error messages per stage
    "errors": {},
    # Config
    "retrieval_k": 10,
    "rerank_k": 5,
    "use_numeric_checker": True,
    "use_calibrated_abstention": True,
}


def init_state():
    """Initialize all session state keys with defaults if not already set."""
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_query_state():
    """Clear pipeline outputs when a new question is asked."""
    query_keys = [
        "pipeline_ran", "retrieved_chunks", "reranked_chunks", "context",
        "original_answer", "claims", "evidence_results", "verification_results",
        "correction_changes", "corrected_answer", "reliability_report",
        "latency", "errors"
    ]
    for k in query_keys:
        st.session_state[k] = DEFAULTS[k]


def reset_document_state():
    """Clear all state when a new document is uploaded."""
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
