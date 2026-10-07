"""
Pure helper functions for the TRUSTRAG Streamlit UI.
All functions here are pure/deterministic and do NOT import streamlit,
making them easy to unit test without launching a server.
"""
from typing import Dict, Any, List, Optional


SUPPORTED_EXTENSIONS = {"pdf"}


def is_supported_file(filename: str) -> bool:
    """Return True if the file extension is supported by TRUSTRAG ingestion."""
    if not filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in SUPPORTED_EXTENSIONS


def get_status_icon(status: str) -> str:
    """Return a unicode icon for a verification status label."""
    return {
        "SUPPORTED": "✅",
        "UNSUPPORTED": "⚠️",
        "CONTRADICTED": "❌",
    }.get(status.upper(), "❓")


def get_correction_label(action: Optional[str]) -> str:
    """Return a human-readable correction label."""
    return {
        "PRESERVED": "✅ Preserved",
        "CORRECTED": "🔧 Corrected",
        "REMOVED_OR_QUALIFIED": "🚫 Removed / Qualified",
    }.get(action or "", "— Not processed")


def format_support_rate(rate: float, supported: int, total: int) -> str:
    """Format claim support rate as a human-readable sentence."""
    pct = f"{rate * 100:.1f}%"
    return f"{pct} — {supported} of {total} extracted claims were supported by the supplied evidence."


def format_problematic_rate(rate: float, problematic: int, total: int) -> str:
    pct = f"{rate * 100:.1f}%"
    return f"{pct} — {problematic} of {total} claims were unsupported or contradicted."


def format_score_note(score: float) -> str:
    """Format a model confidence score with a disclaimer."""
    return f"{score:.4f}  *(model confidence score — not a calibrated probability)*"


def summarise_metrics(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert raw ReportMetrics dict into display-ready strings.
    """
    total = metrics.get("total_claims", 0)
    sup = metrics.get("supported_claims", 0)
    uns = metrics.get("unsupported_claims", 0)
    con = metrics.get("contradicted_claims", 0)
    csr = metrics.get("claim_support_rate", 0.0)
    pcr = metrics.get("problematic_claim_rate", 0.0)
    prob = uns + con

    return {
        "total": str(total),
        "supported": str(sup),
        "unsupported": str(uns),
        "contradicted": str(con),
        "support_rate": format_support_rate(csr, sup, total),
        "problematic_rate": format_problematic_rate(pcr, prob, total),
        "preserved": str(metrics.get("preserved_claims", 0)),
        "corrected": str(metrics.get("corrected_claims", 0)),
        "removed_qualified": str(metrics.get("removed_or_qualified_claims", 0)),
    }


def build_evidence_chain(evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build an ordered, display-ready evidence chain list from raw evidence dicts.
    Each entry has: evidence_id, document, pages, chunk_id, text.
    """
    chain = []
    for ev in evidence:
        page_s = ev.get("page_start", ev.get("pages", ""))
        page_e = ev.get("page_end", "")
        if page_e and page_e != page_s:
            pages = f"{page_s}–{page_e}"
        else:
            pages = str(page_s)

        chain.append({
            "evidence_id": ev.get("evidence_id", ""),
            "document": ev.get("document", ev.get("document_name", "")),
            "pages": pages,
            "chunk_id": ev.get("chunk_id", ""),
            "text": ev.get("text", ""),
            "faiss_score": ev.get("faiss_score", ev.get("retrieval_score", 0.0)),
            "rerank_score": ev.get("rerank_score", 0.0),
        })
    return chain
