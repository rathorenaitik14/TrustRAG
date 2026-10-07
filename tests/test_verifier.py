import pytest
from typing import Dict, Any
from verification.verifier import BaseVerifier

class MockVerifier(BaseVerifier):
    def __init__(self, responses: list):
        super().__init__(entailment_threshold=0.5, contradiction_threshold=0.5)
        self.responses = responses
        self.call_count = 0

    def verify_single_evidence(self, claim_text: str, evidence_text: str) -> Dict[str, Any]:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp

def test_supported_aggregation():
    verifier = MockVerifier([{"status": "UNSUPPORTED", "score": 0.4}, {"status": "SUPPORTED", "score": 0.9}])
    result = verifier.verify({
        "claim_id": "C1",
        "claim_text": "A",
        "evidence": [{"evidence_id": "E1", "text": "1"}, {"evidence_id": "E2", "text": "2"}]
    })
    assert result["status"] == "SUPPORTED"
    assert result["score"] == 0.9
    assert result["evidence_ids"] == ["E2"]

def test_contradicted_aggregation():
    # Contradiction is stronger (0.8 > 0.6)
    verifier = MockVerifier([{"status": "SUPPORTED", "score": 0.6}, {"status": "CONTRADICTED", "score": 0.8}])
    result = verifier.verify({
        "claim_id": "C1",
        "claim_text": "A",
        "evidence": [{"evidence_id": "E1", "text": "1"}, {"evidence_id": "E2", "text": "2"}]
    })
    assert result["status"] == "CONTRADICTED"
    assert result["score"] == 0.8
    assert result["evidence_ids"] == ["E2"]

def test_unsupported_aggregation():
    verifier = MockVerifier([{"status": "UNSUPPORTED", "score": 0.4}, {"status": "UNSUPPORTED", "score": 0.4}])
    result = verifier.verify({
        "claim_id": "C1",
        "claim_text": "A",
        "evidence": [{"evidence_id": "E1", "text": "1"}, {"evidence_id": "E2", "text": "2"}]
    })
    assert result["status"] == "UNSUPPORTED"

def test_empty_evidence():
    verifier = MockVerifier([])
    result = verifier.verify({
        "claim_id": "C1",
        "claim_text": "A",
        "evidence": []
    })
    assert result["status"] == "UNSUPPORTED"
    assert result["reason"] == "No evidence provided."

def test_evidence_level_scores_preserved():
    verifier = MockVerifier([{"status": "SUPPORTED", "score": 0.85}])
    result = verifier.verify({
        "claim_id": "C1", "claim_text": "A",
        "evidence": [{"evidence_id": "E1", "text": "1"}]
    })
    assert result["evidence_details"][0]["score"] == 0.85
    assert result["evidence_details"][0]["evidence_id"] == "E1"

def test_claim_id_text_preserved():
    verifier = MockVerifier([{"status": "UNSUPPORTED", "score": 0.0}])
    result = verifier.verify({
        "claim_id": "C123", "claim_text": "Preserve me",
        "evidence": [{"evidence_id": "E1", "text": "1"}]
    })
    assert result["claim_id"] == "C123"
    assert result["claim_text"] == "Preserve me"

def test_original_evidence_not_mutated():
    verifier = MockVerifier([{"status": "SUPPORTED", "score": 0.9}])
    ev = [{"evidence_id": "E1", "text": "1"}]
    claim_record = {"claim_id": "C1", "claim_text": "A", "evidence": ev}
    ev_copy = list(ev)
    
    verifier.verify(claim_record)
    assert claim_record["evidence"] == ev_copy

def test_configurable_thresholds():
    verifier = MockVerifier([{"status": "SUPPORTED", "score": 0.6}])
    verifier.entailment_threshold = 0.9  # Set high threshold
    # Wait, the MockVerifier itself returns SUPPORTED. The base verifier only checks threshold if we are aggregating. 
    # Actually, BaseVerifier aggregates based on the scores returned by verify_single_evidence.
    # So if max_entail_score (0.6) < 0.9, it becomes UNSUPPORTED.
    result = verifier.verify({
        "claim_id": "C1", "claim_text": "A",
        "evidence": [{"evidence_id": "E1", "text": "1"}]
    })
    assert result["status"] == "UNSUPPORTED"
