import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from retrieval.embeddings import embed_query
from retrieval.faiss_store import load_faiss_index, search_index

# Configuration defaults
DEFAULT_TOP_K = 10
MAX_TOP_K = 50
MIN_TOP_K = 1

class RAGRetriever:
    """
    Reusable RAG retriever that orchestrates question embedding,
    FAISS similarity search, and evidence resolution.
    """
    def __init__(
        self, 
        faiss_index_path: str, 
        metadata_path: str, 
        chunks_jsonl_path: str,
        default_top_k: int = DEFAULT_TOP_K
    ):
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        self.chunks_jsonl_path = chunks_jsonl_path
        self.default_top_k = default_top_k
        
        # Load components
        self._load_components()

    def _load_components(self):
        """Loads FAISS index, metadata, and chunk text mappings safely."""
        if not os.path.exists(self.faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.faiss_index_path}")
            
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
            
        if not os.path.exists(self.chunks_jsonl_path):
            raise FileNotFoundError(f"Chunks JSONL file not found at {self.chunks_jsonl_path}")

        try:
            self.index = load_faiss_index(self.faiss_index_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index. It may be corrupt or incompatible: {e}")

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.index_mapping = self.metadata.get("index_mapping", {})
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}")

        # Validate vector counts
        if len(self.index_mapping) != self.index.ntotal:
            raise ValueError(
                f"Mismatch between FAISS vectors ({self.index.ntotal}) "
                f"and metadata records ({len(self.index_mapping)})"
            )

        # Load chunk dictionary
        self.chunks_dict = {}
        try:
            with open(self.chunks_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.chunks_dict[data["chunk_id"]] = data
        except Exception as e:
            raise RuntimeError(f"Failed to load chunks data: {e}")

    def retrieve(self, question: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Given a natural language question, retrieve the most relevant evidence chunks.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
            
        k = k if k is not None else self.default_top_k
        
        if k < MIN_TOP_K or k > MAX_TOP_K:
            raise ValueError(f"k must be between {MIN_TOP_K} and {MAX_TOP_K}. Got: {k}")

        # 1. Embed question using existing embeddings module
        try:
            query_vector = embed_query(question)
        except Exception as e:
            raise RuntimeError(f"Failed to embed question: {e}")
            
        if query_vector.shape[0] != self.index.d:
            raise ValueError(
                f"Query dimension mismatch. Expected {self.index.d}, got {query_vector.shape[0]}"
            )

        # 2. Search FAISS index
        try:
            results = search_index(
                index=self.index,
                query_vector=query_vector,
                k=k,
                metadata_mapping=self.index_mapping,
                chunks_dict=self.chunks_dict
            )
        except Exception as e:
            raise RuntimeError(f"FAISS search failed: {e}")
            
        return results
