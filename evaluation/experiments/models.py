from typing import TypedDict, List, Dict, Any

class RetrievalMetricsResult(TypedDict):
    recall_1: float
    recall_3: float
    recall_5: float
    mrr: float

class VerificationMetricsResult(TypedDict):
    accuracy: float
    precision_supported: float
    recall_supported: float
    f1_supported: float
    precision_unsupported: float
    recall_unsupported: float
    f1_unsupported: float
    precision_contradicted: float
    recall_contradicted: float
    f1_contradicted: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: Dict[str, Dict[str, int]]
    # Binary problematic claims
    binary_precision: float
    binary_recall: float
    binary_f1: float
    binary_accuracy: float

class CorrectionMetricsResult(TypedDict):
    total_problematic_claims: int
    successful_corrections: int
    correction_success_rate: float
    total_unsupported_claims: int
    unsupported_claim_removal_rate: float
    total_contradicted_claims: int
    contradicted_correction_rate: float
    residual_problematic_claims: int

class LatencyMetrics(TypedDict):
    mean_latency: float
    median_latency: float
    samples: int
