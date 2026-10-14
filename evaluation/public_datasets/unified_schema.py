"""
Unified schema for all public benchmark datasets.
Maps external datasets to a common format that plugs into the existing
BenchmarkRunner and VerificationMetricsCalculator without changes.
"""
from typing import TypedDict, List, Optional


class UnifiedBenchmarkCase(TypedDict):
    """
    Canonical benchmark case used across RAGTruth, HaluEval, and the
    original 15-case homemade dataset. Fields mirror BenchmarkCase so
    the existing BenchmarkRunner.run() method works without modification.
    """
    case_id: str                    # Unique identifier  e.g. "ragtruth-0042"
    source_dataset: str             # "ragtruth" | "halueval" | "homemade"
    task_type: str                  # "qa" | "summarization" | "data2text"
    question: str                   # The query / instruction
    context: str                    # Retrieved passage(s) used as evidence
    answer: str                     # The LLM answer to verify
    ground_truth_label: str         # "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED"
    difficulty: str                 # "easy" | "medium" | "hard"
    notes: str                      # Free-form annotation notes

    # Compatibility fields for BenchmarkRunner
    document_id: str
    document_name: str
    grounded_answer: str
    injected_answer: str
    original_claims: List[str]
    injected_claim: str
    evidence_reference: str
    expected_correction_action: str  # "PRESERVED" | "REMOVED_OR_QUALIFIED" | "CORRECTED"
