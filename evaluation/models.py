from typing import TypedDict, List, Optional, Any

class ReportMetrics(TypedDict):
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    contradicted_claims: int
    preserved_claims: int
    corrected_claims: int
    removed_or_qualified_claims: int
    claim_support_rate: float
    problematic_claim_rate: float

class ReportEvidence(TypedDict):
    evidence_id: str
    document: str
    pages: str
    chunk_id: str
    text: str

class ReportClaim(TypedDict):
    claim_id: str
    claim_text: str
    verification_status: str
    verification_score: float
    evidence_ids: List[str]
    evidence: List[ReportEvidence]
    correction_action: Optional[str]
    correction_status_reason: Optional[str]

class ReliabilityReportDict(TypedDict):
    question: str
    original_answer: str
    corrected_answer: str
    claims: List[ReportClaim]
    metrics: ReportMetrics
    summary: str
