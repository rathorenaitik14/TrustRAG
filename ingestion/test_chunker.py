import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.models import PageRecord
from ingestion.chunker import chunk_pages, estimate_tokens

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.test_chunker <path_to_processed_jsonl>")
        sys.exit(1)
        
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)
        
    # 1. Load existing processed PDF page records
    pages = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pages.append(PageRecord.model_validate_json(line))
                
    if not pages:
        print("No page records found.")
        sys.exit(1)
        
    # 2. Chunk them
    chunks = chunk_pages(pages)
    
    # 3. Print stats
    print(f"Total Pages: {len(pages)}")
    print(f"Total Chunks: {len(chunks)}\n")
    
    for i, chunk in enumerate(chunks[:5]):  # show first 5
        words = estimate_tokens(chunk.text)
        print(f"--- Chunk {chunk.chunk_index} ({chunk.chunk_id}) ---")
        print(f"Page Range: {chunk.page_start} - {chunk.page_end}")
        print(f"Word/Token Count: {words}")
        print(f"Preview (first 150 chars):\n{chunk.text[:150].encode('ascii', 'replace').decode('ascii')}...\n")
        
    # Save to data/processed
    output_file = input_file.parent / f"{pages[0].document_name}_chunks.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + '\n')
            
    print(f"Saved {len(chunks)} chunks to {output_file}")

if __name__ == "__main__":
    main()
