import pytest
from evaluation.benchmark.metrics import BenchmarkMetricsCalculator

def get_dummy_case(case_id, label):
    return {
        "case_id": case_id,
        "ground_truth_label": label
    }

def test_perfect_accuracy():
    cases = [
        get_dummy_case("1", "SUPPORTED"),
        get_dummy_case("2", "UNSUPPORTED"),
        get_dummy_case("3", "CONTRADICTED")
    ]
    predictions = [
        {"case_id": "1", "verification_status": "SUPPORTED", "correction_action": "PRESERVED"},
        {"case_id": "2", "verification_status": "UNSUPPORTED", "correction_action": "REMOVED_OR_QUALIFIED"},
        {"case_id": "3", "verification_status": "CONTRADICTED", "correction_action": "CORRECTED"}
    ]
    calc = BenchmarkMetricsCalculator(cases, predictions)
    res = calc.calculate()
    assert res["verification_accuracy"] == 1.0
    assert res["macro_precision"] == 1.0
    assert res["macro_recall"] == 1.0
    assert res["macro_f1"] == 1.0
    assert res["correction_success_rate"] == 1.0
    assert res["unsupported_claim_removal_rate"] == 1.0

def test_imperfect_accuracy():
    cases = [
        get_dummy_case("1", "SUPPORTED"),
        get_dummy_case("2", "UNSUPPORTED"),
        get_dummy_case("3", "CONTRADICTED")
    ]
    predictions = [
        {"case_id": "1", "verification_status": "SUPPORTED"},
        {"case_id": "2", "verification_status": "SUPPORTED"}, # False positive for supported, False negative for unsupported
        {"case_id": "3", "verification_status": "CONTRADICTED"}
    ]
    calc = BenchmarkMetricsCalculator(cases, predictions)
    res = calc.calculate()
    # Accuracy is 2/3
    assert abs(res["verification_accuracy"] - 0.6666) < 0.001
    
    # Supported: TP=1, FP=1, FN=0 -> Precision=0.5, Recall=1.0, F1=0.666
    assert res["precision_supported"] == 0.5
    assert res["recall_supported"] == 1.0
    
    # Unsupported: TP=0, FP=0, FN=1 -> Precision=0, Recall=0, F1=0
    assert res["precision_unsupported"] == 0.0
    assert res["recall_unsupported"] == 0.0
    
    # Contradicted: TP=1, FP=0, FN=0 -> Precision=1.0, Recall=1.0, F1=1.0
    assert res["precision_contradicted"] == 1.0
    assert res["recall_contradicted"] == 1.0

def test_zero_edge_case():
    calc = BenchmarkMetricsCalculator([], [])
    res = calc.calculate()
    assert res["verification_accuracy"] == 0.0
    assert res["macro_precision"] == 0.0
    assert res["correction_success_rate"] == 0.0

def test_correction_metrics():
    cases = [
        get_dummy_case("1", "UNSUPPORTED"),
        get_dummy_case("2", "CONTRADICTED")
    ]
    predictions = [
        {"case_id": "1", "verification_status": "UNSUPPORTED", "correction_action": "PRESERVED"}, # Failed to remove
        {"case_id": "2", "verification_status": "CONTRADICTED", "correction_action": "CORRECTED"} # Successfully corrected
    ]
    calc = BenchmarkMetricsCalculator(cases, predictions)
    res = calc.calculate()
    assert res["total_problematic_claims"] == 2
    assert res["successfully_corrected"] == 1
    assert res["successfully_removed_or_qualified"] == 0
    assert res["correction_success_rate"] == 0.5
    assert res["unsupported_claims"] == 1
    assert res["unsupported_claim_removal_rate"] == 0.0
