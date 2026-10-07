import pytest
from pathlib import Path
import fitz
from ingestion.pdf_loader import load_pdf, normalize_text

@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    
    # Page 1: Normal text with numbers and dates
    page1 = doc.new_page()
    page1.insert_text((50, 50), "TRUSTRAG Project\nDate: 2023-10-25\n\nSales increased by 45.5% this quarter.")
    
    # Page 2: Empty page
    page2 = doc.new_page()
    
    # Page 3: Technical terminology
    page3 = doc.new_page()
    page3.insert_text((50, 50), "Vector databases like FAISS use L2 distance for embeddings.")
    
    doc.save(pdf_path)
    doc.close()
    return pdf_path

def test_load_valid_pdf(sample_pdf):
    records = load_pdf(str(sample_pdf))
    # Page 2 was empty, so we expect 2 records
    assert len(records) == 2
    
    assert records[0].page_number == 1
    assert "TRUSTRAG Project" in records[0].text
    assert "45.5%" in records[0].text
    
    assert records[1].page_number == 3
    assert "FAISS" in records[1].text
    
    # Check stable document ID
    assert records[0].document_id == records[1].document_id
    assert records[0].source_path == str(sample_pdf.resolve())

def test_missing_pdf():
    with pytest.raises(FileNotFoundError):
        load_pdf("non_existent_file.pdf")

def test_invalid_extension(tmp_path):
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("Hello")
    with pytest.raises(ValueError, match="Expected a PDF"):
        load_pdf(str(txt_path))

def test_normalize_text():
    raw_text = "Hello    World \n\n\n This is 100%   \t true."
    normalized = normalize_text(raw_text)
    assert normalized == "Hello World\nThis is 100% true."
