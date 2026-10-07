import pytest
from evaluation.experiments.ablation import AblationStudy
from unittest.mock import MagicMock

def test_ablation_configurations():
    # Provide mocks for retriever, reranker, verifier
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_verifier = MagicMock()
    
    def mock_ev_ret_factory(ret, rer):
        mock_ev = MagicMock()
        mock_ev.retrieve_for_claims.return_value = [{"claim_id": "mock"}]
        return mock_ev
        
    study = AblationStudy(
        retriever=mock_retriever,
        reranker=mock_reranker,
        verifier=mock_verifier,
        evidence_retriever_factory=mock_ev_ret_factory
    )
    
    ret_res = study.run_retrieval_ablation()
    assert "A1_faiss_only" in ret_res
    assert "A2_faiss_rerank" in ret_res
    
    mock_verifier.verify.return_value = {"status": "SUPPORTED"}
    ver_res = study.run_verification_ablation()
    assert "A3_verification" in ver_res
    assert "predictions" in ver_res
    
    cor_res = study.run_correction_ablation(ver_res["predictions"])
    assert "A4_correction" in cor_res
