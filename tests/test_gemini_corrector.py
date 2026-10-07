import pytest
import json
from unittest.mock import MagicMock
from correction.gemini_corrector import GeminiCorrector

def get_mock_client(return_json: str):
    mock = MagicMock()
    mock.generate.return_value = return_json
    return mock

@pytest.fixture
def mock_claims():
    return [{"claim_id": "C1", "claim_text": "A", "claim_type": "factual"}]

@pytest.fixture
def mock_verifications():
    return [{
        "claim_id": "C1",
        "claim_text": "A",
        "status": "SUPPORTED",
        "score": 0.9,
        "evidence_ids": ["E1"],
        "reason": "OK",
        "evidence_details": [{"evidence_id": "E1", "status": "SUPPORTED", "score": 0.9}]
    }]

@pytest.fixture
def mock_evidence():
    return [{
        "claim_id": "C1",
        "claim_text": "A",
        "evidence": [{"evidence_id": "E1", "text": "A is true."}]
    }]

def test_supported_claim_preservation(mock_claims, mock_verifications, mock_evidence):
    # Test 1, 7, 8, 17
    mock_resp = json.dumps({
        "original_answer": "A",
        "corrected_answer": "A",
        "changes": [{"claim_id": "C1", "status": "SUPPORTED", "action": "PRESERVED"}]
    })
    corrector = GeminiCorrector(client=get_mock_client(mock_resp))
    res = corrector.correct("A", mock_claims, mock_evidence, mock_verifications)
    assert res["corrected_answer"] == "A"
    assert res["original_answer"] == "A"
    assert res["changes"][0]["action"] == "PRESERVED"
    assert res["changes"][0]["claim_id"] == "C1"

def test_contradicted_claim_correction(mock_claims, mock_verifications, mock_evidence):
    # Test 2
    mock_verifications[0]["status"] = "CONTRADICTED"
    mock_resp = json.dumps({
        "original_answer": "A",
        "corrected_answer": "B",
        "changes": [{"claim_id": "C1", "status": "CONTRADICTED", "action": "CORRECTED"}]
    })
    corrector = GeminiCorrector(client=get_mock_client(mock_resp))
    res = corrector.correct("A", mock_claims, mock_evidence, mock_verifications)
    assert res["changes"][0]["action"] == "CORRECTED"

def test_unsupported_claim_removal(mock_claims, mock_verifications, mock_evidence):
    # Test 3
    mock_verifications[0]["status"] = "UNSUPPORTED"
    mock_resp = json.dumps({
        "original_answer": "A",
        "corrected_answer": "Not established.",
        "changes": [{"claim_id": "C1", "status": "UNSUPPORTED", "action": "REMOVED_OR_QUALIFIED"}]
    })
    corrector = GeminiCorrector(client=get_mock_client(mock_resp))
    res = corrector.correct("A", mock_claims, mock_evidence, mock_verifications)
    assert res["changes"][0]["action"] == "REMOVED_OR_QUALIFIED"

def test_invalid_json_handling(mock_claims, mock_verifications, mock_evidence):
    # Test 11
    corrector = GeminiCorrector(client=get_mock_client("This is not JSON"))
    with pytest.raises(ValueError, match="Failed to parse LLM response"):
        corrector.correct("A", mock_claims, mock_evidence, mock_verifications)

def test_missing_fields_handling(mock_claims, mock_verifications, mock_evidence):
    # Test 11
    corrector = GeminiCorrector(client=get_mock_client('{"original_answer": "A"}'))
    with pytest.raises(ValueError, match="missing required fields"):
        corrector.correct("A", mock_claims, mock_evidence, mock_verifications)

def test_invalid_action_labels_fallback(mock_claims, mock_verifications, mock_evidence):
    # Test 18
    mock_resp = json.dumps({
        "original_answer": "A",
        "corrected_answer": "B",
        "changes": [{"claim_id": "C1", "status": "SUPPORTED", "action": "INVALID_ACTION"}]
    })
    corrector = GeminiCorrector(client=get_mock_client(mock_resp))
    res = corrector.correct("A", mock_claims, mock_evidence, mock_verifications)
    assert res["changes"][0]["action"] == "REMOVED_OR_QUALIFIED"

def test_original_objects_not_mutated(mock_claims, mock_verifications, mock_evidence):
    # Test 15
    import copy
    v_copy = copy.deepcopy(mock_verifications)
    mock_resp = json.dumps({
        "original_answer": "A",
        "corrected_answer": "A",
        "changes": [{"claim_id": "C1", "status": "SUPPORTED", "action": "PRESERVED"}]
    })
    corrector = GeminiCorrector(client=get_mock_client(mock_resp))
    corrector.correct("A", mock_claims, mock_evidence, mock_verifications)
    assert mock_verifications == v_copy

def test_empty_claims_and_evidence():
    # Test 5, 6
    mock_resp = json.dumps({
        "original_answer": "A",
        "corrected_answer": "A",
        "changes": []
    })
    client = get_mock_client(mock_resp)
    corrector = GeminiCorrector(client=client)
    corrector.correct("A", [], [], [])
    
    # Verify prompt was constructed without issues
    call_kwargs = client.generate.call_args.kwargs
    assert "CLAIMS:" in call_kwargs["user_prompt"]

def test_prompt_construction(mock_claims, mock_verifications, mock_evidence):
    # Test 9, 10, 12, 14, 16
    mock_resp = json.dumps({
        "original_answer": "A",
        "corrected_answer": "A",
        "changes": []
    })
    client = get_mock_client(mock_resp)
    corrector = GeminiCorrector(client=client)
    corrector.correct("A", mock_claims, mock_evidence, mock_verifications)
    
    sys_prompt = client.generate.call_args.kwargs["system_prompt"]
    user_prompt = client.generate.call_args.kwargs["user_prompt"]
    
    assert "You must not introduce facts" in sys_prompt
    assert "remove the unsupported assertion" in sys_prompt.lower()
    assert "C1" in user_prompt
    assert "Status: SUPPORTED" in user_prompt
    assert "E1: A is true." in user_prompt

def test_api_credentials_not_in_output(mock_claims, mock_verifications, mock_evidence):
    # Test 13
    mock_resp = json.dumps({
        "original_answer": "A",
        "corrected_answer": "A",
        "changes": []
    })
    client = get_mock_client(mock_resp)
    corrector = GeminiCorrector(client=client)
    res = corrector.correct("A", mock_claims, mock_evidence, mock_verifications)
    # Output is structured dict, doesn't contain LLM configs
    assert "api_key" not in res
