import pytest
from unittest.mock import MagicMock
from claims.models import Claim
from claims.evidence_retriever import EvidenceRetriever

def get_mock_retriever():
    mock = MagicMock()
    mock.retrieve.return_value = [
        {
            "chunk_id": "chunk1",
            "score": 0.8,
            "text": "text1",
            "metadata": {"document_name": "doc1", "page_start": 1}
        },
        {
            "chunk_id": "chunk2",
            "score": 0.7,
            "text": "text2",
            "metadata": {"document_name": "doc2", "page_start": 2}
        }
    ]
    return mock

def get_mock_reranker():
    mock = MagicMock()
    def mock_rerank(query, candidates, top_k):
        res = []
        for i, c in enumerate(candidates):
            if i >= top_k: break
            c_copy = dict(c)
            c_copy["rerank_score"] = 9.0 - i
            res.append(c_copy)
        return res
    mock.rerank.side_effect = mock_rerank
    return mock

@pytest.fixture
def evidence_retriever():
    return EvidenceRetriever(get_mock_retriever(), get_mock_reranker())

def test_single_claim_retrieval(evidence_retriever):
    claim = {"claim_id": "C1", "claim_text": "A fact.", "claim_type": "factual"}
    res = evidence_retriever.retrieve_for_claim(claim)
    assert res["claim_id"] == "C1"
    assert len(res["evidence"]) == 2
    assert res["evidence"][0]["text"] == "text1"

def test_multiple_claim_retrieval(evidence_retriever):
    claims = [
        {"claim_id": "C1", "claim_text": "F1", "claim_type": "factual"},
        {"claim_id": "C2", "claim_text": "F2", "claim_type": "factual"}
    ]
    res = evidence_retriever.retrieve_for_claims(claims)
    assert len(res) == 2
    assert res[0]["claim_id"] == "C1"
    assert res[1]["claim_id"] == "C2"

def test_same_components_reused():
    mock_ret = get_mock_retriever()
    mock_rerank = get_mock_reranker()
    er = EvidenceRetriever(mock_ret, mock_rerank)
    er.retrieve_for_claim({"claim_id": "C1", "claim_text": "A", "claim_type": "factual"})
    # retriever and reranker should have been called exactly once
    assert mock_ret.retrieve.call_count == 1
    assert mock_rerank.rerank.call_count == 1

def test_retrieval_k_respected(evidence_retriever):
    claim = {"claim_id": "C1", "claim_text": "A", "claim_type": "factual"}
    evidence_retriever.retrieve_for_claim(claim, retrieval_k=15)
    evidence_retriever.retriever.retrieve.assert_called_with("A", k=15)

def test_evidence_top_k_respected(evidence_retriever):
    claim = {"claim_id": "C1", "claim_text": "A", "claim_type": "factual"}
    # The reranker mock respects top_k
    res = evidence_retriever.retrieve_for_claim(claim, evidence_top_k=1)
    assert len(res["evidence"]) == 1

def test_deterministic_ids_c1(evidence_retriever):
    claim = {"claim_id": "C1", "claim_text": "A", "claim_type": "factual"}
    res = evidence_retriever.retrieve_for_claim(claim)
    assert res["evidence"][0]["evidence_id"] == "C1-E1"
    assert res["evidence"][1]["evidence_id"] == "C1-E2"

def test_deterministic_ids_c2(evidence_retriever):
    claim = {"claim_id": "C2", "claim_text": "B", "claim_type": "factual"}
    res = evidence_retriever.retrieve_for_claim(claim)
    assert res["evidence"][0]["evidence_id"] == "C2-E1"

def test_claim_text_preserved(evidence_retriever):
    claim = {"claim_id": "C1", "claim_text": "Preserve me", "claim_type": "factual"}
    res = evidence_retriever.retrieve_for_claim(claim)
    assert res["claim_text"] == "Preserve me"

def test_scores_preserved(evidence_retriever):
    claim = {"claim_id": "C1", "claim_text": "A", "claim_type": "factual"}
    res = evidence_retriever.retrieve_for_claim(claim)
    assert res["evidence"][0]["faiss_score"] == 0.8
    assert res["evidence"][0]["rerank_score"] == 9.0

def test_metadata_preserved(evidence_retriever):
    claim = {"claim_id": "C1", "claim_text": "A", "claim_type": "factual"}
    res = evidence_retriever.retrieve_for_claim(claim)
    assert res["evidence"][0]["document_name"] == "doc1"
    assert res["evidence"][0]["page_start"] == 1

def test_empty_claim_handling(evidence_retriever):
    res = evidence_retriever.retrieve_for_claim({"claim_id": "C1", "claim_text": "", "claim_type": "factual"})
    assert len(res["evidence"]) == 0

def test_empty_retrieval_handling(evidence_retriever):
    evidence_retriever.retriever.retrieve.return_value = []
    res = evidence_retriever.retrieve_for_claim({"claim_id": "C1", "claim_text": "A", "claim_type": "factual"})
    assert len(res["evidence"]) == 0

def test_non_factual_claim_handling(evidence_retriever):
    res = evidence_retriever.retrieve_for_claim({"claim_id": "C1", "claim_text": "Opinion", "claim_type": "other"})
    assert len(res["evidence"]) == 0

def test_original_objects_not_mutated(evidence_retriever):
    claim = {"claim_id": "C1", "claim_text": "A", "claim_type": "factual"}
    claim_copy = dict(claim)
    evidence_retriever.retrieve_for_claim(claim)
    assert claim == claim_copy

def test_no_verification_labels(evidence_retriever):
    claim = {"claim_id": "C1", "claim_text": "A", "claim_type": "factual"}
    res = evidence_retriever.retrieve_for_claim(claim)
    # Ensure no 'verification', 'unsupported', 'false' is in output keys
    for e in res["evidence"]:
        assert "verification" not in e
        assert "is_true" not in e
