import json
import numpy as np
from pathlib import Path
from ingestion.models import ChunkRecord
from retrieval.embeddings import embed_chunks

def load_chunks(jsonl_path: Path):
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                chunks.append(ChunkRecord(**data))
    return chunks

def process_file(jsonl_path: Path):
    print(f"Processing chunks from {jsonl_path.name}...")
    chunks = load_chunks(jsonl_path)
    
    if not chunks:
        print("No chunks found.")
        return
        
    print(f"Loaded {len(chunks)} chunks.")
    
    embeddings, metadata = embed_chunks(chunks)
    
    print(f"Generated embeddings of shape {embeddings.shape}")
    
    # Paths for saving
    npy_path = jsonl_path.with_name(f"{jsonl_path.stem}_embeddings.npy")
    meta_path = jsonl_path.with_name(f"{jsonl_path.stem}_embeddings_metadata.json")
    
    np.save(npy_path, embeddings)
    print(f"Saved embeddings to {npy_path}")
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {meta_path}")

if __name__ == "__main__":
    base_dir = Path("data/processed")
    for file in base_dir.glob("*_chunks.jsonl"):
        process_file(file)
