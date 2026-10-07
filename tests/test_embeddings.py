import pytest
import numpy as np
from ingestion.models import ChunkRecord
from retrieval.embeddings import (
    load_embedding_model,
    embed_chunks,
    embed_query,
    embed_texts
)

@pytest.fixture
def sample_chunks():
    return [
        ChunkRecord(
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="doc.pdf",
            text="Data mining is the process of discovering patterns in large data sets.",
            page_start=1,
            page_end=1,
            source_path="/path",
            chunk_index=0
        ),
        ChunkRecord(
            chunk_id="chunk_2",
            document_id="doc_1",
            document_name="doc.pdf",
            text="Machine learning algorithms are used for predictions.",
            page_start=2,
            page_end=2,
            source_path="/path",
            chunk_index=1
        )
    ]

def test_model_loading():
    model = load_embedding_model()
    assert model is not None
    assert type(model).__name__ == "SentenceTransformer"

def test_embed_non_empty_text():
    vec = embed_query("Hello world")
    assert isinstance(vec, np.ndarray)
    assert len(vec.shape) == 1
    assert vec.shape[0] > 0

def test_embed_empty_text():
    with pytest.raises(ValueError, match="Cannot embed an empty query"):
        embed_query("")
        
    with pytest.raises(ValueError, match="Cannot embed an empty query"):
        embed_query("   ")

def test_embed_chunks_metadata_and_dimensions(sample_chunks):
    vectors, metadata = embed_chunks(sample_chunks)
    
    assert vectors.shape[0] == 2
    dim = vectors.shape[1]
    assert dim > 0
    
    assert metadata["dimension"] == dim
    assert metadata["num_vectors"] == 2
    assert metadata["index_mapping"]["0"] == "chunk_1"
    assert metadata["index_mapping"]["1"] == "chunk_2"
    assert metadata["normalized"] is True

def test_query_dimension_matches_chunk_dimension(sample_chunks):
    vectors, _ = embed_chunks(sample_chunks)
    query_vec = embed_query("Testing dimension match")
    
    assert vectors.shape[1] == query_vec.shape[0]

def test_deterministic_embedding():
    vec1 = embed_query("Deterministic test string")
    vec2 = embed_query("Deterministic test string")
    
    assert np.allclose(vec1, vec2, atol=1e-6)
    
def test_empty_chunks_list():
    with pytest.raises(ValueError, match="No chunks provided"):
        embed_chunks([])
