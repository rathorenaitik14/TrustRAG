import os
import json
import numpy as np
from pathlib import Path

from retrieval.faiss_store import build_faiss_index, save_faiss_index, load_faiss_index, search_index
from retrieval.embeddings import embed_query
from ingestion.models import ChunkRecord

def load_chunks(jsonl_path: Path):
    chunks = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                chunk = ChunkRecord(**data)
                chunks[chunk.chunk_id] = chunk
    return chunks

def main():
    base_dir = Path("data/processed")
    
    # 1. Identify the files
    jsonl_files = list(base_dir.glob("*_chunks.jsonl"))
    if not jsonl_files:
        print("No chunk files found.")
        return
        
    target_jsonl = next((f for f in jsonl_files if "DEDV" in f.name), jsonl_files[0])
    npy_path = target_jsonl.with_name(f"{target_jsonl.stem}_embeddings.npy")
    meta_path = target_jsonl.with_name(f"{target_jsonl.stem}_embeddings_metadata.json")
    
    if not npy_path.exists() or not meta_path.exists():
        print("Embeddings or metadata not found. Run embedding generation first.")
        return
        
    # 2. Load data
    print(f"Loading chunks from {target_jsonl.name}...")
    chunk_dict = load_chunks(target_jsonl)
    
    print("Loading embeddings and metadata...")
    embeddings = np.load(npy_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    index_mapping = metadata["index_mapping"]
    
    # 3. Build & Save FAISS Index
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    
    index_path = target_jsonl.with_name(f"{target_jsonl.stem}_faiss.index")
    save_faiss_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path} with {index.ntotal} vectors.")
    
    # 4. Search Testing
    print("\nLoading FAISS index for search test...")
    loaded_index = load_faiss_index(str(index_path))
    
    test_queries = [
        "What is the goal of data mining?",
        "What is data preprocessing?",
        "What is KDD?",
        "What is the data engineering lifecycle?",
        "What are measures of central tendency?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: '{query}' ---")
        query_vec = embed_query(query)
        
        results = search_index(
            loaded_index, 
            query_vec, 
            k=3, 
            metadata_mapping=index_mapping, 
            chunks_dict=chunk_dict
        )
        
        for result in results:
            rank = result["rank"]
            score = result["score"]
            chunk_id = result["chunk_id"]
            doc_name = result["document_name"]
            page_start = result["page_start"]
            page_end = result["page_end"]
            text = result["text"]
            
            print(f"\nRank {rank} (Score: {score:.4f})")
            print(f"Chunk ID: {chunk_id}")
            print(f"Document: {doc_name}, Page: {page_start}-{page_end}")
            
            preview = text.replace('\n', ' ')[:150] + "..." if len(text) > 150 else text.replace('\n', ' ')
            print(f"Preview: {preview}")

if __name__ == "__main__":
    main()
