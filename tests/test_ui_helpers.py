"""
Unit tests for TRUSTRAG UI helper functions (components.py).

These tests are fully deterministic — they do NOT launch a Streamlit server
and do NOT call the Gemini API.
"""
import pytest
from ui.components import (
    is_supported_file,
    get_status_icon,
    get_correction_label,
    format_score_note,
    summarise_metrics,
    build_evidence_chain,
)


# ── is_supported_file ─────────────────────────────────────────────────────────

class TestIsSupportedFile:
    def test_pdf_supported(self):
        assert is_supported_file("document.pdf") is True
        assert is_supported_file("REPORT.PDF") is True

    def test_unsupported_types(self):
        assert is_supported_file("file.docx") is False
        assert is_supported_file("data.csv") is False
        assert is_supported_file("sheet.xlsx") is False
        assert is_supported_file("image.png") is False

    def test_empty_string(self):
        assert is_supported_file("") is False

    def test_no_extension(self):
        assert is_supported_file("somefile") is False


# ── get_status_icon ───────────────────────────────────────────────────────────

class TestGetStatusIcon:
    def test_supported(self):
        assert get_status_icon("SUPPORTED") == "✅"

    def test_unsupported(self):
        assert get_status_icon("UNSUPPORTED") == "⚠️"

    def test_contradicted(self):
        assert get_status_icon("CONTRADICTED") == "❌"

    def test_unknown(self):
        assert get_status_icon("UNKNOWN") == "❓"

    def test_case_insensitive(self):
        assert get_status_icon("supported") == "✅"


# ── get_correction_label ──────────────────────────────────────────────────────

class TestGetCorrectionLabel:
    def test_preserved(self):
        assert "Preserved" in get_correction_label("PRESERVED")

    def test_corrected(self):
        assert "Corrected" in get_correction_label("CORRECTED")

    def test_removed(self):
        assert "Removed" in get_correction_label("REMOVED_OR_QUALIFIED")

    def test_none(self):
        label = get_correction_label(None)
        assert "Not processed" in label


# ── format_score_note ─────────────────────────────────────────────────────────

class TestFormatScoreNote:
    def test_contains_score(self):
        note = format_score_note(0.9802)
        assert "0.9802" in note

    def test_contains_disclaimer(self):
        note = format_score_note(0.5)
        assert "not a calibrated probability" in note


# ── summarise_metrics ─────────────────────────────────────────────────────────

class TestSummariseMetrics:
    def get_metrics(self):
        return {
            "total_claims": 3,
            "supported_claims": 2,
            "unsupported_claims": 1,
            "contradicted_claims": 0,
            "preserved_claims": 2,
            "corrected_claims": 0,
            "removed_or_qualified_claims": 1,
            "claim_support_rate": 2 / 3,
            "problematic_claim_rate": 1 / 3,
        }

    def test_total(self):
        s = summarise_metrics(self.get_metrics())
        assert s["total"] == "3"

    def test_supported(self):
        s = summarise_metrics(self.get_metrics())
        assert s["supported"] == "2"

    def test_support_rate_contains_fraction(self):
        s = summarise_metrics(self.get_metrics())
        assert "2 of 3" in s["support_rate"]

    def test_zero_claims_edge_case(self):
        metrics = {
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "contradicted_claims": 0,
            "preserved_claims": 0,
            "corrected_claims": 0,
            "removed_or_qualified_claims": 0,
            "claim_support_rate": 0.0,
            "problematic_claim_rate": 0.0,
        }
        s = summarise_metrics(metrics)
        assert s["total"] == "0"


# ── build_evidence_chain ──────────────────────────────────────────────────────

class TestBuildEvidenceChain:
    def test_basic_chain(self):
        evidence = [
            {
                "evidence_id": "C1-E1",
                "document_name": "doc.pdf",
                "page_start": 3,
                "page_end": 5,
                "chunk_id": "abc123",
                "text": "Some evidence text.",
                "faiss_score": 0.85,
                "rerank_score": 0.92,
            }
        ]
        chain = build_evidence_chain(evidence)
        assert len(chain) == 1
        assert chain[0]["evidence_id"] == "C1-E1"
        assert "3" in chain[0]["pages"]
        assert "5" in chain[0]["pages"]
        assert chain[0]["text"] == "Some evidence text."

    def test_same_page_start_end(self):
        ev = [{"evidence_id": "X", "document_name": "d", "page_start": 7, "page_end": 7,
               "chunk_id": "c", "text": "t"}]
        chain = build_evidence_chain(ev)
        assert chain[0]["pages"] == "7"

    def test_empty_list(self):
        assert build_evidence_chain([]) == []
