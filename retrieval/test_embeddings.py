import sys
import json
import logging
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.models import ChunkRecord
from retrieval.embeddings import embed_chunks, embed_query

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m retrieval.test_embeddings <path_to_chunks_jsonl>")
        sys.exit(1)
        
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: Chunk file not found: {input_file}")
        sys.exit(1)
        
    # 1. Load chunks
    chunks = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(ChunkRecord.model_validate_json(line))
                
    if not chunks:
        print("No chunks found in the file.")
        sys.exit(1)
        
    print(f"Loaded {len(chunks)} chunks.")
    
    # 2. Embed chunks
    print("Generating embeddings for chunks...")
    vectors, metadata = embed_chunks(chunks, normalize=True)
    
    print(f"Embedding model: {metadata['model_name']}")
    print(f"Vector dimension: {metadata['dimension']}")
    print(f"Vector array shape: {vectors.shape}")
    
    # Save the embeddings and metadata
    output_dir = input_file.parent
    vectors_file = output_dir / f"{input_file.stem}_embeddings.npy"
    metadata_file = output_dir / f"{input_file.stem}_embeddings_metadata.json"
    
    np.save(vectors_file, vectors)
    with open(metadata_file, "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved vectors to {vectors_file} and metadata to {metadata_file}")
    
    # 3. Test queries
    test_queries = [
        "What is the purpose of data mining?",
        "What is KDD?",
        "What is data preprocessing?",
        "What is the data engineering lifecycle?",
        "What are measures of central tendency?"
    ]
    
    print("\n" + "="*50)
    print("TESTING SEMANTIC SIMILARITY")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_vec = embed_query(query, normalize=True)
        
        similarities = []
        for i, chunk_vec in enumerate(vectors):
            sim = cosine_similarity(query_vec, chunk_vec)
            similarities.append((i, sim))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        top_k = 3
        for rank in range(top_k):
            if rank >= len(similarities):
                break
            idx, score = similarities[rank]
            chunk_id = metadata["index_mapping"][str(idx)]
            
            matching_chunk = next(c for c in chunks if c.chunk_id == chunk_id)
            
            print(f"  Rank {rank+1} (Score: {score:.4f})")
            print(f"    Document: {matching_chunk.document_name}, Pages: {matching_chunk.page_start}-{matching_chunk.page_end}")
            print(f"    Preview: {matching_chunk.text[:150].encode('ascii', 'replace').decode('ascii')}...")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
