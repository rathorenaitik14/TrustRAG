import pytest
from evaluation.experiments.retrieval_metrics import RetrievalMetricsCalculator

def test_retrieval_metrics():
    gt = {"1": "foo", "2": "bar", "3": "baz"}
    preds = {
        "1": [{"text": "not correct"}, {"text": "here is foo"}], # rank 2 -> R@3, R@5, 1/2 MRR
        "2": [{"text": "bar is here"}],                      # rank 1 -> R@1, R@3, R@5, 1 MRR
        "3": [{"text": "not it"}]                            # no hit -> 0
    }
    
    calc = RetrievalMetricsCalculator()
    res = calc.calculate(gt, preds)
    
    assert res["recall_1"] == 1/3
    assert res["recall_3"] == 2/3
    assert res["recall_5"] == 2/3
    assert abs(res["mrr"] - (0.5 + 1.0) / 3) < 0.001

def test_retrieval_metrics_empty():
    calc = RetrievalMetricsCalculator()
    res = calc.calculate({}, {})
    assert res["recall_1"] == 0.0
