import pytest
from ingestion.models import PageRecord
from ingestion.chunker import chunk_pages, generate_chunk_id

@pytest.fixture
def sample_pages():
    return [
        PageRecord(
            document_id="doc123",
            document_name="test.pdf",
            page_number=1,
            text="First paragraph on page 1.\n\nSecond paragraph on page 1. This has more detail to it.",
            source_path="/fake/test.pdf"
        ),
        PageRecord(
            document_id="doc123",
            document_name="test.pdf",
            page_number=2,
            text="First paragraph on page 2.\n\n" + "Very long paragraph. " * 100,  # simulate long paragraph
            source_path="/fake/test.pdf"
        )
    ]

def test_chunking_basic(sample_pages):
    chunks = chunk_pages(sample_pages)
    
    assert len(chunks) > 0
    # Every chunk should not be empty
    for chunk in chunks:
        assert len(chunk.text.strip()) > 0
        
    # Check valid source metadata
    assert chunks[0].document_id == "doc123"
    assert chunks[0].document_name == "test.pdf"
    assert chunks[0].source_path == "/fake/test.pdf"

def test_deterministic_id(sample_pages):
    chunks1 = chunk_pages(sample_pages)
    chunks2 = chunk_pages(sample_pages)
    
    # Running the same input twice produces the same chunks/IDs
    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        
def test_page_numbers_preserved(sample_pages):
    chunks = chunk_pages(sample_pages)
    
    # The first chunk should start at page 1
    assert chunks[0].page_start == 1
    
    # The last chunk should end at page 2
    assert chunks[-1].page_end == 2
    
def test_long_paragraph_handling():
    long_text = "Word " * 1000
    pages = [PageRecord(
        document_id="doc_long",
        document_name="long.pdf",
        page_number=1,
        text=long_text,
        source_path="/fake/long.pdf"
    )]
    
    chunks = chunk_pages(pages)
    
    # Should split into at least 2 chunks since MAX_CHUNK_TOKENS is 700
    assert len(chunks) > 1
    
    # Reassembling should roughly contain all words
    total_words = sum(len(c.text.split()) for c in chunks)
    # Total words will be >= 1000 due to potential overlap
    assert total_words >= 1000
