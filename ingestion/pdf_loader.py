import os
import hashlib
import logging
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List

from ingestion.models import PageRecord

logger = logging.getLogger(__name__)

def generate_document_id(file_path: str) -> str:
    """Generate a stable, deterministic ID based on the file absolute path."""
    path_obj = Path(file_path).resolve()
    # Use MD5 of the absolute path as a stable identifier
    return hashlib.md5(str(path_obj).encode('utf-8')).hexdigest()

def normalize_text(text: str) -> str:
    """
    Normalize unnecessary whitespace while preserving meaningful characters 
    like numbers, dates, percentages, and technical terms.
    """
    if not text:
        return ""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n{2,}', '\n', text)
    # Replace multiple spaces/tabs with a single space, but keep newlines intact
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()

def load_pdf(path: str) -> List[PageRecord]:
    """
    Load a PDF and extract page-level records.
    """
    file_path = Path(path)
    
    # 1. Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # 2. Validate it is a PDF
    if file_path.suffix.lower() != '.pdf':
        raise ValueError(f"Expected a PDF file, got {file_path.suffix}")

    document_id = generate_document_id(str(file_path))
    document_name = file_path.name
    records = []
    
    try:
        # 3. Open the PDF using PyMuPDF
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {path}: {e}")
        raise RuntimeError(f"Could not open PDF file {path}") from e

    # 4. Iterate through every page
    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            # 5. Extract text page by page
            text = page.get_text()
            
            # 6. Normalize unnecessary whitespace
            normalized_text = normalize_text(text)
            
            # 9. Skip completely empty pages only if appropriate
            if not normalized_text:
                logger.info(f"Skipping empty page {page_num + 1} in {document_name}")
                continue
                
            # 11. Return structured records
            record = PageRecord(
                document_id=document_id,
                document_name=document_name,
                file_type="pdf",
                page_number=page_num + 1,  # 1-indexed
                text=normalized_text,
                source_path=str(file_path.resolve())
            )
            records.append(record)
            
        except Exception as e:
            # 10. Handle extraction errors cleanly by logging
            logger.error(f"Error extracting text from page {page_num + 1} of {document_name}: {e}")
            continue
            
    doc.close()
    return records
