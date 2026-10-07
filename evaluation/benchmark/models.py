from typing import TypedDict, Optional, List

class BenchmarkCase(TypedDict):
    case_id: str
    document_id: str
    document_name: str
    question: str
    grounded_answer: str
    injected_answer: str
    original_claims: List[str]
    injected_claim: str
    ground_truth_label: str  # SUPPORTED, UNSUPPORTED, CONTRADICTED
    evidence_reference: str
    expected_correction_action: str  # PRESERVED, CORRECTED, REMOVED_OR_QUALIFIED
    difficulty: str
    notes: str

class BenchmarkMetricsResult(TypedDict):
    total_cases: int
    correctly_verified_cases: int
    verification_accuracy: float
    
    # Per class metrics
    precision_supported: float
    recall_supported: float
    f1_supported: float
    
    precision_unsupported: float
    recall_unsupported: float
    f1_unsupported: float
    
    precision_contradicted: float
    recall_contradicted: float
    f1_contradicted: float
    
    # Macro metrics
    macro_precision: float
    macro_recall: float
    macro_f1: float
    
    # Correction metrics
    total_problematic_claims: int
    successfully_corrected: int
    successfully_removed_or_qualified: int
    correction_success_rate: float
    
    # Unsupported specific
    unsupported_claims: int
    removed_or_qualified_unsupported: int
    unsupported_claim_removal_rate: float
