import pytest
from evaluation.experiments.verification_metrics import VerificationMetricsCalculator

def test_verification_metrics():
    preds = [
        {"true_label": "SUPPORTED", "pred_label": "SUPPORTED"},
        {"true_label": "UNSUPPORTED", "pred_label": "CONTRADICTED"},
        {"true_label": "CONTRADICTED", "pred_label": "CONTRADICTED"}
    ]
    calc = VerificationMetricsCalculator()
    res = calc.calculate(preds)
    
    assert res["accuracy"] == 2/3
    assert res["precision_supported"] == 1.0
    assert res["recall_supported"] == 1.0
    
    # Binary:
    # SUP -> SUP (TN)
    # UNSUP -> CON (TP)
    # CON -> CON (TP)
    assert res["binary_accuracy"] == 1.0
    assert res["binary_precision"] == 1.0
    assert res["binary_recall"] == 1.0
    
    assert res["confusion_matrix"]["UNSUPPORTED"]["CONTRADICTED"] == 1
    
def test_verification_empty():
    calc = VerificationMetricsCalculator()
    res = calc.calculate([])
    assert res["accuracy"] == 0.0
