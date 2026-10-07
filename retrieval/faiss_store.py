import os
import faiss
import numpy as np
from typing import Dict, List, Any, Optional

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from a NumPy array of embeddings.
    Assumes embeddings are normalized (L2) and uses Inner Product for cosine similarity.
    """
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Embeddings must be a NumPy array.")
    
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
        
    dimension = embeddings.shape[1]
    
    # Use Inner Product index since our embeddings are normalized
    index = faiss.IndexFlatIP(dimension)
    
    if embeddings.shape[0] > 0:
        index.add(embeddings)
        
    return index

def save_faiss_index(index: faiss.Index, path: str) -> None:
    """Save the FAISS index to disk."""
    faiss.write_index(index, path)

def load_faiss_index(path: str) -> faiss.Index:
    """Load the FAISS index from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found at {path}")
    return faiss.read_index(path)

def search_index(
    index: faiss.Index, 
    query_vector: np.ndarray, 
    k: int = 5,
    metadata_mapping: Optional[Dict[str, Any]] = None,
    chunks_dict: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search the FAISS index for the query vector.
    If metadata_mapping and chunks_dict are provided, returns enriched results.
    """
    if len(query_vector.shape) == 1:
        query_vector = np.expand_dims(query_vector, axis=0)
        
    if query_vector.shape[1] != index.d:
        raise ValueError(f"Query dimension {query_vector.shape[1]} does not match index dimension {index.d}")
        
    if index.ntotal == 0:
        return []
        
    k = min(k, index.ntotal)
    
    scores, indices = index.search(query_vector, k)
    
    results = []
    for rank, (score, vector_id) in enumerate(zip(scores[0], indices[0]), start=1):
        if vector_id == -1:
            continue
            
        result = {
            "rank": rank,
            "vector_id": int(vector_id),
            "score": float(score)
        }
        
        # Enrich with metadata if provided
        if metadata_mapping and chunks_dict:
            chunk_id = metadata_mapping.get(str(vector_id))
            if chunk_id and chunk_id in chunks_dict:
                chunk = chunks_dict[chunk_id]
                # Assuming chunk is a ChunkRecord or dict
                if hasattr(chunk, 'model_dump'):
                    chunk_data = chunk.model_dump()
                elif hasattr(chunk, 'dict'):
                    chunk_data = chunk.dict()
                else:
                    chunk_data = chunk
                    
                result.update({
                    "chunk_id": chunk_id,
                    "document_id": chunk_data.get("document_id"),
                    "document_name": chunk_data.get("document_name"),
                    "page_start": chunk_data.get("page_start"),
                    "page_end": chunk_data.get("page_end"),
                    "section": chunk_data.get("section"),
                    "source_path": chunk_data.get("source_path"),
                    "text": chunk_data.get("text")
                })
                
        results.append(result)
        
    return results
