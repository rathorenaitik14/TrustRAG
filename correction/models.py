from typing import TypedDict, List

class CorrectionChange(TypedDict):
    claim_id: str
    status: str
    action: str

class CorrectionResult(TypedDict):
    original_answer: str
    corrected_answer: str
    changes: List[CorrectionChange]
