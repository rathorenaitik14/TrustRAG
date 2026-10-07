import re
import hashlib
from typing import List
from ingestion.models import PageRecord, ChunkRecord

TARGET_CHUNK_TOKENS = 500
MIN_CHUNK_TOKENS = 300
MAX_CHUNK_TOKENS = 700
OVERLAP_TOKENS = 75

def estimate_tokens(text: str) -> int:
    """Simple token estimation by word count."""
    return len(text.split())

def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Deterministic chunk ID."""
    return hashlib.md5(f"{document_id}_{chunk_index}".encode('utf-8')).hexdigest()

def chunk_pages(pages: List[PageRecord]) -> List[ChunkRecord]:
    if not pages:
        return []
    
    document_id = pages[0].document_id
    document_name = pages[0].document_name
    source_path = pages[0].source_path

    # Extract all paragraphs and tag them with their source page number
    paragraphs = []
    for page in sorted(pages, key=lambda x: x.page_number):
        page_paras = re.split(r'\n{2,}', page.text)
        for p in page_paras:
            p = p.strip()
            if p:
                paragraphs.append({
                    "text": p,
                    "page_number": page.page_number
                })
    
    chunks = []
    chunk_index = 0
    i = 0
    
    while i < len(paragraphs):
        current_chunk_paras = []
        current_tokens = 0
        start_page = paragraphs[i]["page_number"]
        end_page = start_page
        
        j = i
        while j < len(paragraphs):
            para = paragraphs[j]
            para_tokens = estimate_tokens(para["text"])
            
            # Avoid exceeding max tokens unconditionally
            if current_tokens + para_tokens > MAX_CHUNK_TOKENS:
                break
                
            current_chunk_paras.append(para)
            current_tokens += para_tokens
            end_page = para["page_number"]
            j += 1
            
            if current_tokens >= TARGET_CHUNK_TOKENS:
                break
                
        # Handle cases where a single paragraph exceeds the maximum size
        if j == i and current_tokens == 0:
            para = paragraphs[i]
            # Split by sentences approximately
            sentences = re.split(r'(?<=[.!?]) +', para["text"])
            sub_paragraphs = []
            curr_sub = []
            curr_sub_tokens = 0
            
            for sent in sentences:
                sent_tokens = estimate_tokens(sent)
                if curr_sub_tokens + sent_tokens > MAX_CHUNK_TOKENS and curr_sub:
                    sub_paragraphs.append({"text": " ".join(curr_sub), "page_number": para["page_number"]})
                    curr_sub = [sent]
                    curr_sub_tokens = sent_tokens
                else:
                    curr_sub.append(sent)
                    curr_sub_tokens += sent_tokens
                    
            if curr_sub:
                sub_paragraphs.append({"text": " ".join(curr_sub), "page_number": para["page_number"]})
                
            if len(sub_paragraphs) == 1:
                # Still too big (a single massive block of text). Force truncate by words.
                words = sub_paragraphs[0]["text"].split()
                forced_splits = []
                for k in range(0, len(words), MAX_CHUNK_TOKENS):
                    forced_splits.append({
                        "text": " ".join(words[k:k+MAX_CHUNK_TOKENS]),
                        "page_number": para["page_number"]
                    })
                sub_paragraphs = forced_splits
                
            # Replace paragraph in list and do not advance i
            paragraphs[i:i+1] = sub_paragraphs
            continue
            
        # Finalize chunk
        text = "\n\n".join(p["text"] for p in current_chunk_paras)
        chunk = ChunkRecord(
            chunk_id=generate_chunk_id(document_id, chunk_index),
            document_id=document_id,
            document_name=document_name,
            text=text,
            page_start=start_page,
            page_end=end_page,
            source_path=source_path,
            chunk_index=chunk_index
        )
        chunks.append(chunk)
        chunk_index += 1
        
        if j >= len(paragraphs):
            break
            
        # Calculate overlap
        overlap_tokens = 0
        overlap_paras_count = 0
        for p in reversed(current_chunk_paras):
            pt = estimate_tokens(p["text"])
            if overlap_tokens + pt <= OVERLAP_TOKENS:
                overlap_tokens += pt
                overlap_paras_count += 1
            else:
                break
                
        next_i = j - overlap_paras_count
        if next_i <= i:
            next_i = i + 1
        i = next_i

    return chunks
