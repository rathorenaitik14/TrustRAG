import pytest
from context.context_builder import ContextBuilder

@pytest.fixture
def sample_results():
    return [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_A",
            "document_name": "DocA.pdf",
            "page_start": 1,
            "page_end": 2,
            "section": "Intro",
            "source_path": "/data/DocA.pdf",
            "retrieval_rank": 2,
            "retrieval_score": 0.85,
            "rerank_rank": 1,
            "rerank_score": 5.12,
            "text": "This is chunk 1 text."
        },
        {
            "chunk_id": "chunk_2",
            "document_id": "doc_B",
            "document_name": "DocB.pdf",
            "page_start": 5,
            "page_end": 5,
            "section": "Method",
            "source_path": "/data/DocB.pdf",
            "retrieval_rank": 1,
            "retrieval_score": 0.90,
            "rerank_rank": 2,
            "rerank_score": 3.45,
            "text": "This is chunk 2 text."
        }
    ]

def test_initialization():
    builder = ContextBuilder(top_k=3)
    assert builder.top_k == 3
    
    with pytest.raises(ValueError):
        ContextBuilder(top_k=0)

def test_question_preservation(sample_results):
    builder = ContextBuilder()
    res = builder.build_context("What is test?", sample_results)
    assert res["question"] == "What is test?"
    assert "QUESTION:\nWhat is test?" in res["formatted_context"]

def test_top_k_behavior(sample_results):
    builder = ContextBuilder(top_k=1)
    res = builder.build_context("Q", sample_results)
    assert len(res["evidence_list"]) == 1
    assert res["evidence_list"][0]["chunk_id"] == "chunk_1"

def test_evidence_ids_deterministic(sample_results):
    builder = ContextBuilder()
    res = builder.build_context("Q", sample_results)
    assert res["evidence_list"][0]["evidence_id"] == "E1"
    assert res["evidence_list"][1]["evidence_id"] == "E2"
    assert "EVIDENCE E1" in res["formatted_context"]
    assert "EVIDENCE E2" in res["formatted_context"]

def test_metadata_preservation(sample_results):
    builder = ContextBuilder()
    res = builder.build_context("Q", sample_results)
    e1 = res["evidence_list"][0]
    
    assert e1["chunk_id"] == "chunk_1"
    assert e1["document_id"] == "doc_A"
    assert e1["document_name"] == "DocA.pdf"
    assert e1["page_start"] == 1
    assert e1["page_end"] == 2
    assert e1["section"] == "Intro"
    assert e1["source_path"] == "/data/DocA.pdf"

def test_faiss_score_preservation(sample_results):
    builder = ContextBuilder()
    res = builder.build_context("Q", sample_results)
    assert res["evidence_list"][0]["retrieval_rank"] == 2
    assert res["evidence_list"][0]["retrieval_score"] == 0.85

def test_cross_encoder_score_preservation(sample_results):
    builder = ContextBuilder()
    res = builder.build_context("Q", sample_results)
    assert res["evidence_list"][0]["rerank_rank"] == 1
    assert res["evidence_list"][0]["rerank_score"] == 5.12

def test_formatting_of_source_and_page(sample_results):
    builder = ContextBuilder()
    res = builder.build_context("Q", sample_results)
    context = res["formatted_context"]
    assert "Source: DocA.pdf" in context
    assert "Pages: 1–2" in context
    assert "Source: DocB.pdf" in context
    assert "Pages: 5\n" in context

def test_formatting_multiple_chunks(sample_results):
    builder = ContextBuilder()
    res = builder.build_context("Q", sample_results)
    context = res["formatted_context"]
    assert "--------------------------------------------------" in context

def test_empty_retrieval_behavior():
    builder = ContextBuilder()
    res = builder.build_context("Q", [])
    assert len(res["evidence_list"]) == 0
    assert "No evidence provided." in res["formatted_context"]

def test_invalid_retrieval_record_behavior(sample_results):
    # Remove required field
    del sample_results[0]["text"]
    builder = ContextBuilder()
    with pytest.raises(ValueError, match="missing required field: 'text'"):
        builder.build_context("Q", sample_results)

def test_deterministic_output(sample_results):
    builder = ContextBuilder()
    res1 = builder.build_context("Q", sample_results)
    res2 = builder.build_context("Q", sample_results)
    assert res1["formatted_context"] == res2["formatted_context"]

def test_grounding_instructions_present(sample_results):
    builder = ContextBuilder()
    res = builder.build_context("Q", sample_results)
    instructions = "Use only the provided evidence to answer the question."
    assert instructions in res["formatted_context"]

def test_original_input_not_mutated(sample_results):
    original_id = sample_results[0]["chunk_id"]
    builder = ContextBuilder()
    builder.build_context("Q", sample_results)
    # Check that sample_results wasn't changed (e.g. no evidence_id injected directly into it if using deepcopy)
    assert "evidence_id" not in sample_results[0]
    assert sample_results[0]["chunk_id"] == original_id

def test_empty_question(sample_results):
    builder = ContextBuilder()
    with pytest.raises(ValueError, match="Question cannot be empty"):
        builder.build_context("", sample_results)
