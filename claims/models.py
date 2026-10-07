from typing import TypedDict, List

class Claim(TypedDict):
    claim_id: str
    claim_text: str
    claim_type: str

class ClaimExtractionResult(TypedDict):
    claims: List[Claim]

class ClaimEvidence(TypedDict):
    evidence_id: str
    rank: int
    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    section: str
    source_path: str
    faiss_score: float
    rerank_score: float
    text: str

class ClaimEvidenceResult(TypedDict):
    claim_id: str
    claim_text: str
    evidence: List[ClaimEvidence]
