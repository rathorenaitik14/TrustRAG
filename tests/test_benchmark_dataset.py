import pytest
from evaluation.benchmark.dataset import get_benchmark_dataset

def test_benchmark_schema():
    dataset = get_benchmark_dataset()
    for case in dataset:
        assert "case_id" in case
        assert "ground_truth_label" in case
        assert case["ground_truth_label"] in ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]

def test_supported_cases_exist():
    dataset = get_benchmark_dataset()
    supported = [c for c in dataset if c["ground_truth_label"] == "SUPPORTED"]
    assert len(supported) >= 5

def test_unsupported_cases_exist():
    dataset = get_benchmark_dataset()
    unsupported = [c for c in dataset if c["ground_truth_label"] == "UNSUPPORTED"]
    assert len(unsupported) >= 5

def test_contradicted_cases_exist():
    dataset = get_benchmark_dataset()
    contradicted = [c for c in dataset if c["ground_truth_label"] == "CONTRADICTED"]
    assert len(contradicted) >= 5
