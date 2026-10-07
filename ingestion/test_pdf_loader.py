import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path so we can import from ingestion
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pdf_loader import load_pdf

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.test_pdf_loader <path_to_pdf>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    
    try:
        records = load_pdf(pdf_path)
        print(f"Document Name: {Path(pdf_path).name}")
        print(f"Total Pages Extracted: {len(records)}")
        
        for record in records:
            print(f"\n--- Page {record.page_number} ---")
            print(f"Characters: {len(record.text)}")
            print(f"Preview (first 200 chars):\n{record.text[:200].encode('ascii', 'replace').decode('ascii')}...")
            
        # Save to data/processed
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = processed_dir / f"{Path(pdf_path).stem}_extracted.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(record.model_dump_json() + '\n')
                
        print(f"\nSaved extracted records to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
