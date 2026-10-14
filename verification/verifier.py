from typing import List, Dict, Any
from abc import ABC, abstractmethod
from claims.models import ClaimEvidenceResult
from .models import ClaimVerificationResult, EvidenceVerification

class BaseVerifier(ABC):
    """
    Abstract base class for claim verification.
    """
    def __init__(
        self, 
        entailment_threshold: float = 0.5, 
        contradiction_threshold: float = 0.5,
        use_numeric_checker: bool = True
    ):
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.use_numeric_checker = use_numeric_checker
        self.numeric_checker = None
        if use_numeric_checker:
            try:
                from .numeric_checker import NumericContradictionChecker
                self.numeric_checker = NumericContradictionChecker()
            except Exception:
                self.numeric_checker = None

    @abstractmethod
    def verify_single_evidence(self, claim_text: str, evidence_text: str) -> Dict[str, Any]:
        """
        Evaluate a claim against a single piece of evidence.
        Must return a dict containing at minimum 'status' and 'score'.
        """
        pass

    def verify(self, claim_evidence_result: ClaimEvidenceResult) -> ClaimVerificationResult:
        claim_id = claim_evidence_result.get("claim_id", "")
        claim_text = claim_evidence_result.get("claim_text", "")
        evidence_items = claim_evidence_result.get("evidence", [])

        if not evidence_items:
            return {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "status": "UNSUPPORTED",
                "score": 0.0,
                "evidence_ids": [],
                "reason": "No evidence provided.",
                "evidence_details": []
            }

        evidence_details: List[EvidenceVerification] = []
        max_entail_score = -1.0
        max_contra_score = -1.0
        best_entail_ev = None
        best_contra_ev = None

        for ev in evidence_items:
            ev_id = ev.get("evidence_id", "")
            ev_text = ev.get("text", "")
            pred = None
            
            # Predict single evidence
            try:
                pred = self.verify_single_evidence(claim_text, ev_text)
                # Apply 0.08ms analytical Numeric Contradiction Checker
                if self.numeric_checker and pred.get("status") != "CONTRADICTED":
                    pred = self.numeric_checker.override_nli_result(pred, claim_text, ev_text)
                status = pred["status"]
                score = pred["score"]
            except Exception as e:
                # Handle model/processing errors by defaulting to UNSUPPORTED
                status = "UNSUPPORTED"
                score = 0.0

            evidence_details.append({
                "evidence_id": ev_id,
                "status": status,
                "score": score,
                "reason": pred.get("reason", None) if isinstance(pred, dict) else None
            })

            if status == "SUPPORTED" and score > max_entail_score:
                max_entail_score = score
                best_entail_ev = ev_id
            elif status == "CONTRADICTED" and score > max_contra_score:
                max_contra_score = score
                best_contra_ev = ev_id

        # Deterministic aggregation policy
        final_status = "UNSUPPORTED"
        final_score = 0.0
        evidence_ids = []
        reason = ""

        # Condition 1: Strong contradiction
        if max_contra_score >= self.contradiction_threshold and max_contra_score > max_entail_score:
            final_status = "CONTRADICTED"
            final_score = max_contra_score
            evidence_ids = [best_contra_ev] if best_contra_ev else []
            contra_reason = None
            for d in evidence_details:
                if d.get("evidence_id") == best_contra_ev and d.get("reason"):
                    contra_reason = d.get("reason")
                    break
            reason = contra_reason or "The strongest relevant evidence contradicts the claim."
        # Condition 2: Strong entailment
        elif max_entail_score >= self.entailment_threshold:
            final_status = "SUPPORTED"
            final_score = max_entail_score
            evidence_ids = [best_entail_ev] if best_entail_ev else []
            reason = "The strongest relevant evidence supports the claim."
        else:
            final_status = "UNSUPPORTED"
            final_score = max(max_entail_score, max_contra_score, 0.0)
            reason = "Evidence is insufficient to establish either entailment or contradiction."

        return {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "status": final_status,
            "score": final_score,
            "evidence_ids": evidence_ids,
            "reason": reason,
            "evidence_details": evidence_details
        }
