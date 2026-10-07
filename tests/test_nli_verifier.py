import pytest
from unittest.mock import patch, MagicMock
from verification.nli_verifier import NLIVerifier
import numpy as np

@pytest.fixture
def mock_cross_encoder():
    with patch("verification.nli_verifier.CrossEncoder") as mock_ce:
        instance = mock_ce.return_value
        yield instance

def get_nli_verifier(mock_instance):
    # Don't let it load the real model, mock already does it
    return NLIVerifier(model_name="mock-model")

def test_supported_classification(mock_cross_encoder):
    # Logits: [contra, entail, neutral]
    # We want entailment (index 1) to be high
    mock_cross_encoder.predict.return_value = [np.array([-5.0, 5.0, -5.0])]
    verifier = get_nli_verifier(mock_cross_encoder)
    
    res = verifier.verify_single_evidence("Hypothesis", "Premise")
    assert res["status"] == "SUPPORTED"
    assert res["score"] > 0.9

def test_contradicted_classification(mock_cross_encoder):
    # Logits: contra (index 0) is high
    mock_cross_encoder.predict.return_value = [np.array([5.0, -5.0, -5.0])]
    verifier = get_nli_verifier(mock_cross_encoder)
    
    res = verifier.verify_single_evidence("Hypothesis", "Premise")
    assert res["status"] == "CONTRADICTED"
    assert res["score"] > 0.9

def test_unsupported_classification(mock_cross_encoder):
    # Logits: neutral (index 2) is high
    mock_cross_encoder.predict.return_value = [np.array([-5.0, -5.0, 5.0])]
    verifier = get_nli_verifier(mock_cross_encoder)
    
    res = verifier.verify_single_evidence("Hypothesis", "Premise")
    assert res["status"] == "UNSUPPORTED"
    assert res["score"] > 0.9

def test_empty_evidence(mock_cross_encoder):
    verifier = get_nli_verifier(mock_cross_encoder)
    res = verifier.verify_single_evidence("Hypothesis", "")
    assert res["status"] == "UNSUPPORTED"
    assert res["score"] == 0.0

def test_model_initialization_errors():
    with patch("verification.nli_verifier.CrossEncoder", side_effect=Exception("Model load failed")):
        with pytest.raises(Exception, match="Model load failed"):
            NLIVerifier("bad-model")

def test_invalid_nli_output_handling(mock_cross_encoder):
    # E.g. predicting throws an error
    mock_cross_encoder.predict.side_effect = Exception("Inference failed")
    verifier = get_nli_verifier(mock_cross_encoder)
    
    # The base verifier catches this and returns UNSUPPORTED during aggregation.
    # But verify_single_evidence itself raises. Let's test that verify aggregates it safely.
    result = verifier.verify({
        "claim_id": "C1", "claim_text": "A",
        "evidence": [{"evidence_id": "E1", "text": "1"}]
    })
    
    assert result["status"] == "UNSUPPORTED"
    assert result["evidence_details"][0]["status"] == "UNSUPPORTED"
    assert result["evidence_details"][0]["score"] == 0.0

def test_no_crossencoder_score_used(mock_cross_encoder):
    mock_cross_encoder.predict.return_value = [np.array([-5.0, -5.0, 5.0])]
    verifier = get_nli_verifier(mock_cross_encoder)
    
    claim = {
        "claim_id": "C1",
        "claim_text": "Hypothesis",
        "evidence": [
            {
                "evidence_id": "E1",
                "text": "Premise",
                "rerank_score": 9.99  # very high cross encoder score
            }
        ]
    }
    
    result = verifier.verify(claim)
    # Even though rerank_score is 9.99, NLI returns neutral (index 2)
    assert result["status"] == "UNSUPPORTED"
    
def test_very_long_evidence(mock_cross_encoder):
    # Mocks can handle anything, but just to show the API doesn't choke before calling model
    mock_cross_encoder.predict.return_value = [np.array([-5.0, 5.0, -5.0])]
    verifier = get_nli_verifier(mock_cross_encoder)
    long_text = "word " * 10000
    res = verifier.verify_single_evidence("A", long_text)
    assert res["status"] == "SUPPORTED"

def test_duplicate_evidence_handling(mock_cross_encoder):
    mock_cross_encoder.predict.return_value = [np.array([-5.0, 5.0, -5.0])]
    verifier = get_nli_verifier(mock_cross_encoder)
    
    claim = {
        "claim_id": "C1", "claim_text": "A",
        "evidence": [{"evidence_id": "E1", "text": "1"}, {"evidence_id": "E2", "text": "1"}]
    }
    result = verifier.verify(claim)
    # Both are supported, it aggregates fine
    assert result["status"] == "SUPPORTED"
    assert len(result["evidence_details"]) == 2

def test_unsupported_does_not_become_contradicted(mock_cross_encoder):
    # Unrelated text -> neutral -> unsupported
    mock_cross_encoder.predict.return_value = [np.array([-5.0, -5.0, 5.0])]
    verifier = get_nli_verifier(mock_cross_encoder)
    
    claim = {
        "claim_id": "C1", "claim_text": "Claim",
        "evidence": [{"evidence_id": "E1", "text": "Completely unrelated text"}]
    }
    result = verifier.verify(claim)
    assert result["status"] == "UNSUPPORTED"
