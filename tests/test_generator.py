import pytest
from unittest.mock import MagicMock
from generation.llm_client import LLMClient
from generation.generator import LLMGenerator

class MockLLMClient(LLMClient):
    def __init__(self, expected_answer="Mocked Answer"):
        self.expected_answer = expected_answer
        self.last_system_prompt = None
        self.last_user_prompt = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.expected_answer

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def provider(self) -> str:
        return "Mock Provider"

@pytest.fixture
def sample_context():
    return {
        "question": "What is data preprocessing?",
        "evidence_list": [
            {"evidence_id": "E1", "chunk_id": "chunk_A"},
            {"evidence_id": "E2", "chunk_id": "chunk_B"}
        ],
        "formatted_context": "EVIDENCE E1\nText A\n\nEVIDENCE E2\nText B"
    }

def test_generator_initialization():
    client = MockLLMClient()
    generator = LLMGenerator(client=client)
    assert generator.client == client

def test_question_preservation(sample_context):
    client = MockLLMClient()
    generator = LLMGenerator(client=client)
    res = generator.generate_answer(sample_context)
    assert res["question"] == sample_context["question"]

def test_context_is_passed_to_llm(sample_context):
    client = MockLLMClient()
    generator = LLMGenerator(client=client)
    generator.generate_answer(sample_context)
    
    assert "QUESTION:\nWhat is data preprocessing?" in client.last_user_prompt
    assert "EVIDENCE E1\nText A" in client.last_user_prompt

def test_correct_system_prompt_is_constructed(sample_context):
    client = MockLLMClient()
    generator = LLMGenerator(client=client)
    generator.generate_answer(sample_context)
    
    sys_prompt = client.last_system_prompt
    assert "Answer the user's question using ONLY the supplied evidence." in sys_prompt
    assert "If the evidence is insufficient" in sys_prompt
    assert "Do not invent citations or sources." in sys_prompt

def test_generated_answer_returned(sample_context):
    client = MockLLMClient(expected_answer="Data preprocessing is...")
    generator = LLMGenerator(client=client)
    res = generator.generate_answer(sample_context)
    assert res["answer"] == "Data preprocessing is..."

def test_model_and_provider_preserved(sample_context):
    client = MockLLMClient()
    generator = LLMGenerator(client=client)
    res = generator.generate_answer(sample_context)
    assert res["model_name"] == "mock-model"
    assert res["provider"] == "Mock Provider"

def test_evidence_ids_preserved(sample_context):
    client = MockLLMClient()
    generator = LLMGenerator(client=client)
    res = generator.generate_answer(sample_context)
    assert res["evidence_ids"] == ["E1", "E2"]

def test_empty_context_handling():
    client = MockLLMClient()
    generator = LLMGenerator(client=client)
    
    # Missing formatted_context
    bad_context = {"question": "What?", "evidence_list": []}
    with pytest.raises(ValueError, match="missing required formatted_context"):
        generator.generate_answer(bad_context)

def test_missing_question_handling():
    client = MockLLMClient()
    generator = LLMGenerator(client=client)
    
    bad_context = {"question": "", "evidence_list": [], "formatted_context": ""}
    with pytest.raises(ValueError, match="Context is missing a valid 'question'"):
        generator.generate_answer(bad_context)
        
    bad_context2 = {"evidence_list": [], "formatted_context": ""}
    with pytest.raises(ValueError, match="Context is missing a valid 'question'"):
        generator.generate_answer(bad_context2)

def test_llm_client_errors_handled_clearly(sample_context):
    class ErrorClient(MockLLMClient):
        def generate(self, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("API Offline")
            
    generator = LLMGenerator(client=ErrorClient())
    with pytest.raises(RuntimeError, match="API Offline"):
        generator.generate_answer(sample_context)

def test_original_context_not_mutated(sample_context):
    client = MockLLMClient()
    generator = LLMGenerator(client=client)
    
    original_question = sample_context["question"]
    generator.generate_answer(sample_context)
    
    assert sample_context["question"] == original_question
    assert len(sample_context["evidence_list"]) == 2

def test_api_credentials_not_in_output(sample_context, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "SECRET_KEY_123")
    # Even if we used GeminiClient, it wouldn't output the key, but we test the structure
    client = MockLLMClient()
    generator = LLMGenerator(client=client)
    res = generator.generate_answer(sample_context)
    
    output_str = str(res)
    assert "SECRET_KEY_123" not in output_str

def test_deterministic_prompt_construction(sample_context):
    client1 = MockLLMClient()
    gen1 = LLMGenerator(client=client1)
    gen1.generate_answer(sample_context)
    
    client2 = MockLLMClient()
    gen2 = LLMGenerator(client=client2)
    gen2.generate_answer(sample_context)
    
    assert client1.last_system_prompt == client2.last_system_prompt
    assert client1.last_user_prompt == client2.last_user_prompt
