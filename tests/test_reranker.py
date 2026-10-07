import pytest
from unittest.mock import patch, MagicMock
from retrieval.reranker import CrossEncoderReranker

@pytest.fixture
def sample_candidates():
    return [
        {
            "rank": 1,
            "score": 0.8,
            "chunk_id": "chunk_A",
            "document_id": "doc_1",
            "text": "First chunk text"
        },
        {
            "rank": 2,
            "score": 0.7,
            "chunk_id": "chunk_B",
            "document_id": "doc_1",
            "text": "Second chunk text"
        },
        {
            "rank": 3,
            "score": 0.6,
            "chunk_id": "chunk_C",
            "document_id": "doc_1",
            "text": "Third chunk text"
        }
    ]

@patch("retrieval.reranker.CrossEncoder")
def test_reranker_loads_model(mock_ce):
    reranker = CrossEncoderReranker(model_name="dummy/model")
    assert reranker._model is None
    reranker._load_model()
    mock_ce.assert_called_once_with("dummy/model")
    assert reranker._model is not None

@patch("retrieval.reranker.CrossEncoder")
def test_valid_question_and_candidates_sorting(mock_ce, sample_candidates):
    mock_model = MagicMock()
    # Return scores inverted from original rank to prove sorting works
    # Chunk A: 0.1, Chunk B: 0.9, Chunk C: 0.5
    mock_model.predict.return_value = [0.1, 0.9, 0.5]
    mock_ce.return_value = mock_model
    
    reranker = CrossEncoderReranker()
    results = reranker.rerank("What is the test?", sample_candidates, top_k=3)
    
    assert len(results) == 3
    # B should be first, C second, A third
    assert results[0]["chunk_id"] == "chunk_B"
    assert results[1]["chunk_id"] == "chunk_C"
    assert results[2]["chunk_id"] == "chunk_A"
    
    # Check new ranks
    assert results[0]["rerank_rank"] == 1
    assert results[1]["rerank_rank"] == 2
    assert results[2]["rerank_rank"] == 3
    
    # Check original ranks preserved
    assert results[0]["retrieval_rank"] == 2
    assert results[1]["retrieval_rank"] == 3
    assert results[2]["retrieval_rank"] == 1
    
    # Check scores preserved
    assert results[0]["retrieval_score"] == 0.7
    assert results[0]["rerank_score"] == 0.9

@patch("retrieval.reranker.CrossEncoder")
def test_top_k_limits_output(mock_ce, sample_candidates):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9, 0.5]
    mock_ce.return_value = mock_model
    
    reranker = CrossEncoderReranker()
    results = reranker.rerank("Question", sample_candidates, top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk_B"
    assert results[1]["chunk_id"] == "chunk_C"

def test_empty_candidates():
    reranker = CrossEncoderReranker()
    results = reranker.rerank("Question", [], top_k=5)
    assert results == []

def test_empty_question(sample_candidates):
    reranker = CrossEncoderReranker()
    with pytest.raises(ValueError, match="Question cannot be empty"):
        reranker.rerank("", sample_candidates)

def test_invalid_top_k(sample_candidates):
    reranker = CrossEncoderReranker()
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        reranker.rerank("Question", sample_candidates, top_k=0)

def test_missing_text_in_candidates():
    reranker = CrossEncoderReranker()
    bad_candidates = [{"rank": 1, "score": 0.9, "chunk_id": "chunk_A"}]
    with pytest.raises(ValueError, match="missing 'text' field"):
        reranker.rerank("Question", bad_candidates)
