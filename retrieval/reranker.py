import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Configurable defaults
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_TOP_K = 5
DEFAULT_BATCH_SIZE = 16

class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, batch_size: int = DEFAULT_BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        
    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load CrossEncoder model {self.model_name}: {e}")
                
    def rerank(self, question: str, candidates: List[Dict[str, Any]], top_k: int = DEFAULT_RERANK_TOP_K) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate chunks based on their relevance to the question.
        Preserves original metadata and retrieval rank/score.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
            
        if not candidates:
            return []
            
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0. Got: {top_k}")
            
        self._load_model()
        
        # Prepare inputs for the CrossEncoder: list of (query, candidate_text) pairs
        model_inputs = []
        for cand in candidates:
            if "text" not in cand or not cand["text"]:
                raise ValueError("Malformed candidate: missing 'text' field.")
            model_inputs.append((question, cand["text"]))
            
        # Predict relevance scores
        try:
            scores = self._model.predict(model_inputs, batch_size=self.batch_size)
        except Exception as e:
            raise RuntimeError(f"Failed during CrossEncoder prediction: {e}")
            
        # Combine candidates with their new scores
        reranked_candidates = []
        for idx, (cand, score) in enumerate(zip(candidates, scores)):
            # Preserve original retrieval rank and score
            retrieval_rank = cand.get("rank") or (idx + 1)
            retrieval_score = cand.get("score", 0.0)
            
            # Copy candidate to avoid modifying the original list
            enriched_cand = cand.copy()
            
            enriched_cand["retrieval_rank"] = retrieval_rank
            enriched_cand["retrieval_score"] = float(retrieval_score)
            enriched_cand["rerank_score"] = float(score)
            
            # Remove old 'rank' and 'score' to avoid confusion, 
            # we will set 'rank' -> 'rerank_rank' later
            if "rank" in enriched_cand:
                del enriched_cand["rank"]
            if "score" in enriched_cand:
                del enriched_cand["score"]
                
            reranked_candidates.append(enriched_cand)
            
        # Sort candidates descending by rerank_score
        reranked_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Take top_k
        top_candidates = reranked_candidates[:top_k]
        
        # Assign final rerank_rank
        for i, cand in enumerate(top_candidates, start=1):
            cand["rerank_rank"] = i
            
        return top_candidates
