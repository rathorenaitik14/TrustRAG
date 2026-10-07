import os
import json
import pytest
import numpy as np
import tempfile
import faiss

from retrieval.retriever import RAGRetriever

@pytest.fixture
def retriever_setup():
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        faiss_path = os.path.join(tmpdir, "test.index")
        metadata_path = os.path.join(tmpdir, "metadata.json")
        chunks_path = os.path.join(tmpdir, "chunks.jsonl")
        
        # 1. Create a dummy FAISS index
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        # Add 3 dummy vectors
        np.random.seed(42)
        vectors = np.random.rand(3, dimension).astype('float32')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        index.add(vectors)
        faiss.write_index(index, faiss_path)
        
        # 2. Create metadata mapping
        metadata = {
            "index_mapping": {
                "0": "chunk_0",
                "1": "chunk_1",
                "2": "chunk_2"
            },
            "dimension": dimension,
            "num_vectors": 3
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
            
        # 3. Create chunks JSONL
        chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "document_id": "doc_1",
                "document_name": "test.pdf",
                "page_start": 1,
                "page_end": 1,
                "section": "main",
                "source_path": "/path/test.pdf",
                "text": f"This is chunk {i}"
            } for i in range(3)
        ]
        with open(chunks_path, "w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
                
        yield {
            "faiss_path": faiss_path,
            "metadata_path": metadata_path,
            "chunks_path": chunks_path,
            "dimension": dimension
        }

def test_retriever_initialization_success(retriever_setup):
    retriever = RAGRetriever(
        faiss_index_path=retriever_setup["faiss_path"],
        metadata_path=retriever_setup["metadata_path"],
        chunks_jsonl_path=retriever_setup["chunks_path"]
    )
    assert retriever.index.ntotal == 3
    assert len(retriever.chunks_dict) == 3

def test_missing_faiss_index(retriever_setup):
    with pytest.raises(FileNotFoundError, match="FAISS index not found"):
        RAGRetriever(
            faiss_index_path="missing.index",
            metadata_path=retriever_setup["metadata_path"],
            chunks_jsonl_path=retriever_setup["chunks_path"]
        )

def test_missing_metadata(retriever_setup):
    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        RAGRetriever(
            faiss_index_path=retriever_setup["faiss_path"],
            metadata_path="missing.json",
            chunks_jsonl_path=retriever_setup["chunks_path"]
        )

def test_metadata_vector_count_mismatch(retriever_setup):
    # Alter metadata
    with open(retriever_setup["metadata_path"], "w") as f:
        json.dump({"index_mapping": {"0": "chunk_0"}}, f) # Only 1 mapping for 3 vectors
        
    with pytest.raises(ValueError, match="Mismatch between FAISS vectors"):
        RAGRetriever(
            faiss_index_path=retriever_setup["faiss_path"],
            metadata_path=retriever_setup["metadata_path"],
            chunks_jsonl_path=retriever_setup["chunks_path"]
        )

def test_empty_question_rejected(retriever_setup):
    retriever = RAGRetriever(
        faiss_index_path=retriever_setup["faiss_path"],
        metadata_path=retriever_setup["metadata_path"],
        chunks_jsonl_path=retriever_setup["chunks_path"]
    )
    with pytest.raises(ValueError, match="Question cannot be empty"):
        retriever.retrieve("")

def test_invalid_k_rejected(retriever_setup):
    retriever = RAGRetriever(
        faiss_index_path=retriever_setup["faiss_path"],
        metadata_path=retriever_setup["metadata_path"],
        chunks_jsonl_path=retriever_setup["chunks_path"]
    )
    with pytest.raises(ValueError, match="k must be between"):
        retriever.retrieve("Valid question", k=0)
        
    with pytest.raises(ValueError, match="k must be between"):
        retriever.retrieve("Valid question", k=100)

def test_valid_question_returns_results(retriever_setup):
    retriever = RAGRetriever(
        faiss_index_path=retriever_setup["faiss_path"],
        metadata_path=retriever_setup["metadata_path"],
        chunks_jsonl_path=retriever_setup["chunks_path"]
    )
    results = retriever.retrieve("Hello world", k=2)
    
    assert len(results) == 2
    assert "rank" in results[0]
    assert "score" in results[0]
    assert "chunk_id" in results[0]
    assert "text" in results[0]
    
    # Check correct order
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2
    assert results[0]["score"] >= results[1]["score"]

def test_k_larger_than_available_vectors(retriever_setup):
    retriever = RAGRetriever(
        faiss_index_path=retriever_setup["faiss_path"],
        metadata_path=retriever_setup["metadata_path"],
        chunks_jsonl_path=retriever_setup["chunks_path"]
    )
    # Request 10, but only 3 available
    results = retriever.retrieve("Test", k=10)
    assert len(results) == 3

