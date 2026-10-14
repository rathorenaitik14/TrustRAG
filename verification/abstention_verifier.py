"""
Calibrated Abstention Verifier (Novel Contribution - Stage 3)
=============================================================
Adds a 4th verdict class: UNVERIFIABLE (evidence not retrieved or not relevant).
Key insight: UNSUPPORTED in RAG often conflates two distinct failure modes:
  (a) The claim is false / unsupported by an authoritative corpus
  (b) The retrieval subsystem failed to surface relevant evidence

By distinguishing retrieval failure from factual hallucination, TrustRAG
provides calibrated uncertainty estimates and prevents false alarms.
"""
from typing import Dict, Any, List, Optional
import re

UNVERIFIABLE = "UNVERIFIABLE"

class RelevanceScorer:
    """
    Computes a normalized relevance score in [0, 1] between a claim and evidence.
    Uses token overlap and key term match with stopword filtering.
    """
    _STOPWORDS = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "and", "or", "but", "if", "then", "so", "as",
        "that", "this", "these", "those", "it", "its", "what", "which", "who"
    }

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower())
        return [w for w in words if w not in self._STOPWORDS and len(w) > 1]

    def score(self, claim: str, evidence: str) -> float:
        c_tokens = set(self._tokenize(claim))
        e_tokens = set(self._tokenize(evidence))

        if not c_tokens or not e_tokens:
            return 0.0

        # Jaccard overlap on content words
        overlap = len(c_tokens & e_tokens)
        coverage = overlap / len(c_tokens)
        jaccard = overlap / len(c_tokens | e_tokens)

        # Weighted combination favoring claim coverage
        return round(0.7 * coverage + 0.3 * jaccard, 4)


class CalibratedAbstentionVerifier:
    """
    Adds calibrated abstention mode: returns 'UNVERIFIABLE' when
    evidence relevance falls below the threshold.
    """
    def __init__(
        self,
        base_verifier,
        retrieval_relevance_scorer: Optional[RelevanceScorer] = None,
        relevance_threshold: float = 0.30
    ):
        self.verifier = base_verifier
        self.scorer = retrieval_relevance_scorer or RelevanceScorer()
        self.relevance_threshold = relevance_threshold

    def verify_with_abstention(self, claim: str, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Direct claim and evidence list verification."""
        if not evidence_list:
            return {
                "status": "UNVERIFIABLE",
                "reason": "no_evidence_retrieved",
                "score": 0.0
            }

        top_evidence = evidence_list[0]
        ev_text = top_evidence.get("text", "")
        relevance = self.scorer.score(claim, ev_text)

        if relevance < self.relevance_threshold:
            return {
                "status": "UNVERIFIABLE",
                "reason": "evidence_not_relevant",
                "retrieval_score": relevance,
                "score": relevance
            }

        res = self.verifier.verify_single_evidence(claim, ev_text)
        res["retrieval_score"] = relevance
        return res

    def verify(self, claim_evidence_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compatible with ClaimEvidenceResult interface."""
        claim_id = claim_evidence_result.get("claim_id", "")
        claim_text = claim_evidence_result.get("claim_text", "")
        evidence_items = claim_evidence_result.get("evidence", [])

        if not evidence_items:
            return {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "status": "UNVERIFIABLE",
                "score": 0.0,
                "evidence_ids": [],
                "reason": "No evidence retrieved by retriever.",
                "evidence_details": []
            }

        top_evidence = evidence_items[0]
        ev_text = top_evidence.get("text", "")
        relevance = self.scorer.score(claim_text, ev_text)

        if relevance < self.relevance_threshold:
            return {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "status": "UNVERIFIABLE",
                "score": relevance,
                "evidence_ids": [top_evidence.get("evidence_id", "")],
                "reason": f"Top retrieved evidence relevance ({relevance:.2f}) below abstention threshold ({self.relevance_threshold:.2f}).",
                "evidence_details": []
            }

        # Delegate to base verifier
        result = self.verifier.verify(claim_evidence_result)
        result["retrieval_score"] = relevance
        return result
