import json
import numpy as np
from pathlib import Path
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

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (N,)
    # b: (M, N)
    # returns: (M,)
    dot_product = np.dot(b, a)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    # To avoid division by zero
    norm_a = norm_a if norm_a > 0 else 1e-10
    norm_b = np.where(norm_b > 0, norm_b, 1e-10)
    return dot_product / (norm_a * norm_b)

def run_test():
    base_dir = Path("data/processed")
    jsonl_files = list(base_dir.glob("*_chunks.jsonl"))
    
    if not jsonl_files:
        print("No chunk files found.")
        return
        
    # We use the DEDV PDF as requested, or the first one we find
    target_jsonl = next((f for f in jsonl_files if "DEDV" in f.name), jsonl_files[0])
    npy_path = target_jsonl.with_name(f"{target_jsonl.stem}_embeddings.npy")
    meta_path = target_jsonl.with_name(f"{target_jsonl.stem}_embeddings_metadata.json")
    
    if not npy_path.exists() or not meta_path.exists():
        print(f"Embeddings not found for {target_jsonl.name}. Run generation first.")
        return
        
    print(f"Loading chunks from {target_jsonl.name}...")
    chunk_dict = load_chunks(target_jsonl)
    
    print("Loading embeddings and metadata...")
    embeddings = np.load(npy_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    index_mapping = metadata["index_mapping"]
    
    test_queries = [
        "What is the purpose of data mining?",
        "What is KDD?",
        "What is data preprocessing?",
        "What is the data engineering lifecycle?",
        "What are measures of central tendency?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: '{query}' ---")
        query_vec = embed_query(query)
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, embeddings)
        
        # Get top 3 indices
        top_indices = np.argsort(similarities)[::-1][:3]
        
        for rank, idx in enumerate(top_indices, 1):
            score = similarities[idx]
            chunk_id = index_mapping[str(idx)]
            chunk = chunk_dict[chunk_id]
            
            print(f"\nRank {rank} (Score: {score:.4f})")
            print(f"Document: {chunk.document_name}, Page: {chunk.page_start}-{chunk.page_end}")
            # Preview first 150 chars
            preview = chunk.text.replace('\n', ' ')[:150] + "..." if len(chunk.text) > 150 else chunk.text.replace('\n', ' ')
            print(f"Preview: {preview}")

if __name__ == "__main__":
    run_test()
