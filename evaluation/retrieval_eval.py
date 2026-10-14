"""
Retrieval Evaluation Module (Stage 2C)
======================================
Measures Recall@K and MRR@K for Dense (FAISS), Sparse (BM25), and Hybrid (RRF) retrieval.
Demonstrates the empirical retrieval improvements of hybrid search on publication benchmarks.
"""
from typing import List, Tuple, Dict, Any


def recall_at_k(queries_with_gold: List[Tuple[str, str]], retriever, k: int = 5) -> float:
    """
    Computes Recall@K over a list of (query, gold_chunk_id) tuples.

    Parameters
    ----------
    queries_with_gold : List of (query_text, expected_gold_chunk_id)
    retriever         : Any retriever object implementing .retrieve(query, k)
    k                 : Cutoff rank (e.g. 1, 3, 5, 10)

    Returns
    -------
    float : Recall@K in [0.0, 1.0]
    """
    if not queries_with_gold:
        return 0.0

    hits = 0
    for q, gold_chunk_id in queries_with_gold:
        results = retriever.retrieve(q, k=k)
        retrieved_ids = [r.get("chunk_id", "") for r in results]
        if gold_chunk_id in retrieved_ids:
            hits += 1

    return hits / len(queries_with_gold)


def mrr_at_k(queries_with_gold: List[Tuple[str, str]], retriever, k: int = 10) -> float:
    """
    Computes Mean Reciprocal Rank (MRR@K).
    """
    if not queries_with_gold:
        return 0.0

    reciprocal_ranks = []
    for q, gold_chunk_id in queries_with_gold:
        results = retriever.retrieve(q, k=k)
        rank_found = 0
        for rank, r in enumerate(results, start=1):
            if r.get("chunk_id", "") == gold_chunk_id:
                rank_found = rank
                break
        if rank_found > 0:
            reciprocal_ranks.append(1.0 / rank_found)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_retrievers(
    queries_with_gold: List[Tuple[str, str]],
    retrievers_map: Dict[str, Any],
    k_list: List[int] = [1, 3, 5, 10]
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple retrievers (e.g. Dense-only, BM25-only, Hybrid RRF).
    """
    report = {}
    for name, retriever in retrievers_map.items():
        metrics = {}
        for k in k_list:
            rec = recall_at_k(queries_with_gold, retriever, k=k)
            metrics[f"Recall@{k}"] = round(rec, 4)
        metrics["MRR@10"] = round(mrr_at_k(queries_with_gold, retriever, k=10), 4)
        report[name] = metrics
    return report
