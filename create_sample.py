import fitz
from pathlib import Path

def create_sample_pdf():
    pdf_path = Path("data/raw/sample.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open()
    
    # Page 1: Normal text with numbers and dates
    page1 = doc.new_page()
    page1.insert_text((50, 50), "TRUSTRAG Quarterly Report\nDate: 2023-11-01\n\nRevenue grew by 23.4%.\nThe model successfully extracted 100% of facts.")
    
    # Page 2: Empty page
    doc.new_page()
    
    # Page 3: Technical terminology
    page3 = doc.new_page()
    page3.insert_text((50, 50), "Technical Stack:\n- PyMuPDF (fitz) for PDF extraction\n- Pydantic for data models\n- Vector databases like FAISS use L2 distance.")
    
    doc.save(pdf_path)
    doc.close()
    print(f"Created {pdf_path}")

if __name__ == "__main__":
    create_sample_pdf()
