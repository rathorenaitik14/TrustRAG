import os
import pytest
import numpy as np
import faiss
import tempfile
from retrieval.faiss_store import build_faiss_index, save_faiss_index, load_faiss_index, search_index

@pytest.fixture
def mock_embeddings():
    # 5 vectors of dimension 10
    np.random.seed(42)
    vectors = np.random.rand(5, 10).astype('float32')
    # Normalize for cosine similarity simulation
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

@pytest.fixture
def mock_metadata():
    return {str(i): f"chunk_{i}" for i in range(5)}

@pytest.fixture
def mock_chunks():
    return {
        f"chunk_{i}": {
            "chunk_id": f"chunk_{i}",
            "document_id": "doc_1",
            "document_name": "test.pdf",
            "page_start": 1,
            "page_end": 1,
            "section": "main",
            "source_path": "/path/test.pdf",
            "text": f"This is chunk {i}"
        } for i in range(5)
    }

def test_build_faiss_index(mock_embeddings):
    index = build_faiss_index(mock_embeddings)
    assert isinstance(index, faiss.IndexFlatIP)
    assert index.ntotal == mock_embeddings.shape[0]
    assert index.d == mock_embeddings.shape[1]

def test_save_and_load_index(mock_embeddings):
    index = build_faiss_index(mock_embeddings)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
        
    try:
        save_faiss_index(index, path)
        assert os.path.exists(path)
        
        loaded_index = load_faiss_index(path)
        assert loaded_index.ntotal == index.ntotal
        assert loaded_index.d == index.d
    finally:
        if os.path.exists(path):
            os.remove(path)

def test_search_index(mock_embeddings, mock_metadata, mock_chunks):
    index = build_faiss_index(mock_embeddings)
    
    # Query using the exact same first vector
    query_vector = mock_embeddings[0]
    
    results = search_index(
        index, 
        query_vector, 
        k=3, 
        metadata_mapping=mock_metadata, 
        chunks_dict=mock_chunks
    )
    
    assert len(results) == 3
    assert results[0]["rank"] == 1
    assert results[0]["vector_id"] == 0
    assert results[0]["chunk_id"] == "chunk_0"
    assert results[0]["document_id"] == "doc_1"
    assert results[0]["text"] == "This is chunk 0"
    
    # Check score is roughly 1.0 (since it's exactly the same normalized vector)
    assert np.isclose(results[0]["score"], 1.0, atol=1e-5)
    
def test_search_returns_at_most_k(mock_embeddings):
    index = build_faiss_index(mock_embeddings)
    query_vector = mock_embeddings[0]
    
    results = search_index(index, query_vector, k=2)
    assert len(results) == 2
    
    results_too_many = search_index(index, query_vector, k=10)
    assert len(results_too_many) == index.ntotal

def test_invalid_dimension(mock_embeddings):
    index = build_faiss_index(mock_embeddings)
    
    # Query with wrong dimension
    query_vector = np.random.rand(1, 15).astype('float32')
    
    with pytest.raises(ValueError, match="does not match index dimension"):
        search_index(index, query_vector)

def test_empty_index_handled_safely():
    empty_embeddings = np.zeros((0, 10), dtype='float32')
    index = build_faiss_index(empty_embeddings)
    
    assert index.ntotal == 0
    
    query_vector = np.random.rand(1, 10).astype('float32')
    results = search_index(index, query_vector)
    
    assert results == []

def test_load_non_existent_index():
    with pytest.raises(FileNotFoundError):
        load_faiss_index("does_not_exist.faiss")
