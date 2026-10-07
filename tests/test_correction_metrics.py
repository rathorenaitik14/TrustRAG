import pytest
from evaluation.experiments.correction_metrics import CorrectionMetricsCalculator

def test_correction_metrics():
    preds = [
        {"true_label": "UNSUPPORTED", "pred_action": "REMOVED_OR_QUALIFIED"},
        {"true_label": "CONTRADICTED", "pred_action": "PRESERVED"}, # failed to correct
        {"true_label": "CONTRADICTED", "pred_action": "CORRECTED"},
        {"true_label": "SUPPORTED", "pred_action": "PRESERVED"}
    ]
    calc = CorrectionMetricsCalculator()
    res = calc.calculate(preds)
    
    assert res["total_problematic_claims"] == 3
    assert res["successful_corrections"] == 2
    assert abs(res["correction_success_rate"] - 2/3) < 0.001
    assert res["total_unsupported_claims"] == 1
    assert res["unsupported_claim_removal_rate"] == 1.0
    assert res["total_contradicted_claims"] == 2
    assert res["contradicted_correction_rate"] == 0.5
    assert res["residual_problematic_claims"] == 1

def test_correction_empty():
    calc = CorrectionMetricsCalculator()
    res = calc.calculate([])
    assert res["correction_success_rate"] == 0.0
