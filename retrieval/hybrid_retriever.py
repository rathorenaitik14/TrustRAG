"""
Hybrid Retriever (Stage 2C)
===========================
Combines BM25 (lexical) + FAISS (semantic) with Reciprocal Rank Fusion (RRF).
Solves the low recall problem of dense-only retrieval by capturing exact keyword
matches, technical identifiers, and acronyms that embeddings miss.
"""
from typing import List, Dict, Any, Optional
import re
import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Tokenize query and documents for BM25."""
    return re.findall(r'\b\w+\b', text.lower())


class HybridRetriever:
    """Combines BM25 (lexical) + FAISS (semantic) with Reciprocal Rank Fusion."""

    def __init__(
        self,
        faiss_retriever,
        chunks: List[Dict[str, Any]],
        alpha: float = 0.5,
        k_bm25: int = 60,
        rrf_k: int = 60
    ):
        self.faiss_retriever = faiss_retriever
        self.chunks = chunks
        self.chunk_dict = {c["chunk_id"]: c for c in chunks}
        self.alpha = alpha  # weight between dense (1-alpha) and sparse (alpha)
        self.k_bm25 = k_bm25
        self.rrf_k = rrf_k

        # Index corpus for BM25
        tokenized_corpus = [_tokenize(c.get("text", "")) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks combining dense FAISS search and sparse BM25 search.
        """
        if not query or not query.strip():
            return []

        # 1. Dense retrieval (FAISS)
        retrieve_limit = max(k, min(self.k_bm25, len(self.chunks)))
        try:
            dense_results = self.faiss_retriever.retrieve(query, k=retrieve_limit)
        except Exception:
            dense_results = []
        dense_ids = [r["chunk_id"] for r in dense_results if "chunk_id" in r]

        # 2. Sparse retrieval (BM25)
        query_tokens = _tokenize(query)
        if query_tokens:
            bm25_scores = self.bm25.get_scores(query_tokens)
            bm25_top_idx = np.argsort(bm25_scores)[::-1][:retrieve_limit]
            sparse_ids = [self.chunks[i]["chunk_id"] for i in bm25_top_idx]
        else:
            sparse_ids = []

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        for rank, chunk_id in enumerate(dense_ids):
            dense_ranks[chunk_id] = rank + 1
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 - self.alpha) * (1.0 / (self.rrf_k + rank + 1))

        for rank, chunk_id in enumerate(sparse_ids):
            sparse_ranks[chunk_id] = rank + 1
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + self.alpha * (1.0 / (self.rrf_k + rank + 1))

        # Sort by combined RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:k]

        results = []
        for rank, cid in enumerate(sorted_ids):
            chunk = self.chunk_dict.get(cid)
            if chunk:
                item = dict(chunk)
                item["rank"] = rank + 1
                item["rrf_score"] = round(rrf_scores[cid], 5)
                item["dense_rank"] = dense_ranks.get(cid)
                item["sparse_rank"] = sparse_ranks.get(cid)
                results.append(item)

        return results
