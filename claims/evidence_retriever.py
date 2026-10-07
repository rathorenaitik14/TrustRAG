from typing import List, Dict, Any, Optional
from .models import Claim, ClaimEvidence, ClaimEvidenceResult
from retrieval.retriever import RAGRetriever
from retrieval.reranker import CrossEncoderReranker

class EvidenceRetriever:
    """
    Retrieves evidence specifically for individual factual claims.
    """
    def __init__(
        self,
        retriever: RAGRetriever,
        reranker: CrossEncoderReranker,
        default_retrieval_k: int = 10,
        default_evidence_top_k: int = 5
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.default_retrieval_k = default_retrieval_k
        self.default_evidence_top_k = default_evidence_top_k

    def retrieve_for_claim(
        self, 
        claim: Claim, 
        retrieval_k: Optional[int] = None, 
        evidence_top_k: Optional[int] = None
    ) -> ClaimEvidenceResult:
        """
        Retrieves evidence for a single claim.
        """
        ret_k = retrieval_k if retrieval_k is not None else self.default_retrieval_k
        ev_k = evidence_top_k if evidence_top_k is not None else self.default_evidence_top_k
        
        claim_id = claim.get("claim_id", "")
        claim_text = claim.get("claim_text", "").strip()
        claim_type = claim.get("claim_type", "factual")
        
        # If claim is empty or not factual, return empty evidence
        if not claim_text or claim_type != "factual":
            return {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "evidence": []
            }
            
        # 1. Retrieve candidates
        try:
            candidates = self.retriever.retrieve(claim_text, k=ret_k)
        except Exception:
            candidates = []
            
        if not candidates:
            return {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "evidence": []
            }
            
        # 2. Rerank
        try:
            reranked = self.reranker.rerank(claim_text, candidates, top_k=ev_k)
        except Exception:
            # Fallback to original candidates if reranking fails
            reranked = candidates[:ev_k]
            
        # 3. Format Output
        evidence_list: List[ClaimEvidence] = []
        for i, item in enumerate(reranked, 1):
            evidence_id = f"{claim_id}-E{i}"
            
            # Extract fields safely since original dictionaries might vary
            metadata = item.get("metadata", {})
            if not metadata and "document_id" in item:
                # Flat structure
                metadata = item
            
            faiss_score = item.get("score", item.get("faiss_score", 0.0))
            rerank_score = item.get("rerank_score", 0.0)
                
            evidence_list.append({
                "evidence_id": evidence_id,
                "rank": i,
                "chunk_id": item.get("chunk_id", ""),
                "document_id": metadata.get("document_id", ""),
                "document_name": metadata.get("document_name", ""),
                "page_start": metadata.get("page_start", 0),
                "page_end": metadata.get("page_end", 0),
                "section": metadata.get("section", ""),
                "source_path": metadata.get("source_path", ""),
                "faiss_score": float(faiss_score),
                "rerank_score": float(rerank_score),
                "text": item.get("text", "")
            })
            
        return {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "evidence": evidence_list
        }
        
    def retrieve_for_claims(
        self, 
        claims: List[Claim], 
        retrieval_k: Optional[int] = None, 
        evidence_top_k: Optional[int] = None
    ) -> List[ClaimEvidenceResult]:
        """
        Retrieves evidence for multiple claims.
        """
        results = []
        for claim in claims:
            res = self.retrieve_for_claim(claim, retrieval_k, evidence_top_k)
            results.append(res)
        return results
