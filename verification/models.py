from typing import TypedDict, List, Optional

class EvidenceVerification(TypedDict):
    evidence_id: str
    status: str
    score: float
    reason: Optional[str]

class ClaimVerificationResult(TypedDict):
    claim_id: str
    claim_text: str
    status: str
    score: float
    evidence_ids: List[str]
    reason: Optional[str]
    evidence_details: List[EvidenceVerification]
