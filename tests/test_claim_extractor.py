import pytest
import json
from unittest.mock import MagicMock
from claims.claim_extractor import ClaimExtractor
from generation.llm_client import LLMClient

class MockLLMClient(LLMClient):
    def __init__(self, expected_json):
        self.expected_json = expected_json
        self.last_system_prompt = None
        self.last_user_prompt = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.expected_json

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def provider(self) -> str:
        return "Mock Provider"

def test_extract_one_factual_claim():
    mock_json = '{"claims": [{"claim_text": "The sky is blue.", "claim_type": "factual"}]}'
    extractor = ClaimExtractor(client=MockLLMClient(mock_json))
    res = extractor.extract_claims("The sky is blue.")
    assert len(res["claims"]) == 1
    assert res["claims"][0]["claim_text"] == "The sky is blue."

def test_extract_multiple_claims():
    mock_json = '{"claims": [{"claim_text": "A is B.", "claim_type": "factual"}, {"claim_text": "C is D.", "claim_type": "factual"}]}'
    extractor = ClaimExtractor(client=MockLLMClient(mock_json))
    res = extractor.extract_claims("A is B and C is D.")
    assert len(res["claims"]) == 2

def test_split_compound_claims():
    mock_json = '{"claims": [{"claim_text": "Mean is central.", "claim_type": "factual"}, {"claim_text": "Median is central.", "claim_type": "factual"}]}'
    extractor = ClaimExtractor(client=MockLLMClient(mock_json))
    res = extractor.extract_claims("Mean and median are central measures.")
    assert len(res["claims"]) == 2
    assert "Mean is central." in [c["claim_text"] for c in res["claims"]]

def test_preserve_exact_factual_meaning():
    # Tested inherently by mock returning exact text
    mock_json = '{"claims": [{"claim_text": "Exactly 5.2%.", "claim_type": "factual"}]}'
    extractor = ClaimExtractor(client=MockLLMClient(mock_json))
    res = extractor.extract_claims("The rate is exactly 5.2%.")
    assert res["claims"][0]["claim_text"] == "Exactly 5.2%."

def test_deterministic_ids():
    mock_json = '{"claims": [{"claim_id": "X1", "claim_text": "A", "claim_type": "factual"}, {"claim_id": "X2", "claim_text": "B", "claim_type": "factual"}]}'
    extractor = ClaimExtractor(client=MockLLMClient(mock_json))
    res = extractor.extract_claims("A and B.")
    assert res["claims"][0]["claim_id"] == "C1"
    assert res["claims"][1]["claim_id"] == "C2"

def test_empty_answer_handling():
    extractor = ClaimExtractor(client=MockLLMClient("{}"))
    with pytest.raises(ValueError, match="cannot be empty"):
        extractor.extract_claims("")

def test_missing_answer_handling():
    extractor = ClaimExtractor(client=MockLLMClient("{}"))
    with pytest.raises(ValueError, match="cannot be empty"):
        extractor.extract_claims(None)

def test_invalid_json_handling():
    extractor = ClaimExtractor(client=MockLLMClient("This is not JSON"))
    with pytest.raises(ValueError, match="Failed to parse LLM response"):
        extractor.extract_claims("Valid answer")

def test_missing_claims_field_handling():
    extractor = ClaimExtractor(client=MockLLMClient('{"wrong_key": []}'))
    with pytest.raises(ValueError, match="does not contain the required 'claims' key"):
        extractor.extract_claims("Valid answer")

def test_malformed_claim_object_handling():
    extractor = ClaimExtractor(client=MockLLMClient('{"claims": ["not a dict"]}'))
    with pytest.raises(ValueError, match="Malformed claim object"):
        extractor.extract_claims("Valid answer")
        
    extractor2 = ClaimExtractor(client=MockLLMClient('{"claims": [{"wrong": "val"}]}'))
    with pytest.raises(ValueError, match="Missing 'claim_text'"):
        extractor2.extract_claims("Valid answer")

def test_duplicate_claim_handling():
    mock_json = '{"claims": [{"claim_text": "A"}, {"claim_text": "A"}, {"claim_text": "a"}]}'
    extractor = ClaimExtractor(client=MockLLMClient(mock_json))
    res = extractor.extract_claims("A.")
    assert len(res["claims"]) == 1
    assert res["claims"][0]["claim_text"] == "A"

def test_original_answer_not_mutated():
    original = "This is a test."
    extractor = ClaimExtractor(client=MockLLMClient('{"claims": []}'))
    extractor.extract_claims(original)
    assert original == "This is a test."

def test_prompt_construction_deterministic():
    client1 = MockLLMClient('{"claims": []}')
    ex1 = ClaimExtractor(client=client1)
    ex1.extract_claims("Answer", "Question")
    
    client2 = MockLLMClient('{"claims": []}')
    ex2 = ClaimExtractor(client=client2)
    ex2.extract_claims("Answer", "Question")
    
    assert client1.last_system_prompt == client2.last_system_prompt
    assert client1.last_user_prompt == client2.last_user_prompt
    assert "ORIGINAL QUESTION:\nQuestion" in client1.last_user_prompt

def test_api_credentials_not_in_output(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "SECRET123")
    client = MockLLMClient('{"claims": [{"claim_text": "A", "claim_type": "factual"}]}')
    ex = ClaimExtractor(client=client)
    res = ex.extract_claims("A")
    assert "SECRET123" not in str(res)

def test_extractor_does_not_perform_verification():
    # Tested by ensuring the system prompt forbids it
    client = MockLLMClient('{"claims": []}')
    ex = ClaimExtractor(client=client)
    ex.extract_claims("Answer")
    assert "Do NOT perform verification" in client.last_system_prompt

def test_false_claims_extracted():
    # Tested by ensuring the system prompt specifically requests it
    client = MockLLMClient('{"claims": []}')
    ex = ClaimExtractor(client=client)
    ex.extract_claims("The Earth is made of chocolate.")
    assert "extract it exactly as stated, even if it is factually incorrect (e.g., 'the earth is made of chocolate')" in client.last_system_prompt.lower()

def test_non_factual_ignored():
    client = MockLLMClient('{"claims": []}')
    ex = ClaimExtractor(client=client)
    ex.extract_claims("Hello there!")
    assert "Do not extract greetings or conversational filler" in client.last_system_prompt
