import os
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from ingestion.models import ChunkRecord

logger = logging.getLogger(__name__)

# Configurable embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_NORMALIZE = True

_model = None
_model_name = None

def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the SentenceTransformer model and cache it globally."""
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model
    
    try:
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        return _model
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise RuntimeError(f"Embedding model loading failed: {e}") from e

def embed_texts(texts: List[str], normalize: bool = DEFAULT_NORMALIZE) -> np.ndarray:
    """Core function to embed a list of raw text strings."""
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")
    
    model = load_embedding_model()
    
    # Handle empty text correctly (warn and replace with space to avoid model crashing)
    safe_texts = []
    for t in texts:
        if not t or not t.strip():
            logger.warning("Empty text encountered in embed_texts. Using a single space.")
            safe_texts.append(" ")
        else:
            safe_texts.append(t)
            
    try:
        # Generate embeddings
        embeddings = model.encode(safe_texts, normalize_embeddings=normalize, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise RuntimeError(f"Embedding generation failed: {e}") from e

def embed_chunks(chunks: List[ChunkRecord], normalize: bool = DEFAULT_NORMALIZE) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Embed a list of ChunkRecords and return vectors + metadata mapping."""
    if not chunks:
        raise ValueError("No chunks provided for embedding.")
        
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts, normalize=normalize)
    
    # Create deterministic mapping: vector position (index) -> chunk_id
    index_to_chunk_id = {str(i): chunk.chunk_id for i, chunk in enumerate(chunks)}
    
    metadata = {
        "model_name": _model_name,
        "dimension": int(embeddings.shape[1]),
        "normalized": normalize,
        "num_vectors": len(chunks),
        "index_mapping": index_to_chunk_id
    }
    
    return embeddings, metadata

def embed_query(query: str, normalize: bool = DEFAULT_NORMALIZE) -> np.ndarray:
    """Embed a single user query."""
    if not query or not query.strip():
        raise ValueError("Cannot embed an empty query.")
    
    embeddings = embed_texts([query], normalize=normalize)
    return embeddings[0]
