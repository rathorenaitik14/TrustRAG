"""Unit tests for publication modules (no GPU/API needed)."""
import pytest
import numpy as np
from verification.numeric_checker import NumericContradictionChecker
from evaluation.stats import macro_f1, bootstrap_ci, mcnemar_test, all_metrics_with_ci
from rank_bm25 import BM25Okapi
from evaluation.public_datasets.unified_schema import UnifiedBenchmarkCase
from evaluation.public_datasets.ragtruth_loader import RAGTruthLoader
from evaluation.public_datasets.halueval_loader import HaluEvalLoader


def test_numeric_contradiction_percentage():
    checker = NumericContradictionChecker()
    r = checker.check(
        claim="By 2025, 0 percent of data will be in the cloud.",
        evidence="By 2025, 49 percent of data will be stored in public cloud."
    )
    assert r["verdict"] == "CONTRADICTED", f"Expected CONTRADICTED, got {r['verdict']}"


def test_numeric_consistency():
    checker = NumericContradictionChecker()
    r = checker.check(
        claim="By 2025, 49 percent of data will be stored in cloud.",
        evidence="By 2025, 49 percent of data will be stored in public cloud environments."
    )
    assert r["verdict"] == "CONSISTENT", f"Expected CONSISTENT, got {r['verdict']}"


def test_numeric_unit_contradiction():
    checker = NumericContradictionChecker()
    r = checker.check(
        claim="IoT devices will create 1 byte of data by 2025.",
        evidence="90ZB of data will be created on IoT devices by 2025."
    )
    assert r["verdict"] == "CONTRADICTED", f"Expected CONTRADICTED, got {r['verdict']}"


def test_numeric_no_numbers():
    checker = NumericContradictionChecker()
    r = checker.check("Data mining is useful.", "Data mining helps with KDD.")
    assert r["verdict"] == "NO_NUMBERS"


def test_stats_metrics_and_ci():
    y_t = ["SUPPORTED"] * 30 + ["UNSUPPORTED"] * 30 + ["CONTRADICTED"] * 30
    y_p = (
        ["SUPPORTED"] * 25 + ["UNSUPPORTED"] * 5
        + ["UNSUPPORTED"] * 25 + ["CONTRADICTED"] * 5
        + ["CONTRADICTED"] * 30
    )

    f1 = macro_f1(y_t, y_p)
    assert 0.6 < f1 < 1.0, f"Unexpected F1 value: {f1}"

    pt, lo, hi = bootstrap_ci(y_t, y_p, macro_f1, n_bootstrap=100)
    assert lo <= pt <= hi, f"CI bounds wrong: {lo} <= {pt} <= {hi}"

    sig = mcnemar_test(y_t, y_t, y_t)
    assert not sig["significant_at_p05"], "Identical predictions should not be significant"

    full = all_metrics_with_ci(y_t, y_p, n_bootstrap=100)
    assert "macro_f1" in full
    assert "per_class" in full
    assert "SUPPORTED" in full["per_class"]


def test_bm25_lexical_retrieval():
    chunks = [
        {"chunk_id": "c1", "text": "By 2025, 49 percent of data will be stored in public cloud."},
        {"chunk_id": "c2", "text": "Data mining provides efficient tools for KDD."},
        {"chunk_id": "c3", "text": "90ZB of data will be created on IoT devices by 2025."},
    ]
    corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores("cloud storage percentage".split())
    best_idx = int(np.argmax(scores))
    assert best_idx == 0, f"Expected chunk 0 to rank first for 'cloud', got {best_idx}"


def test_dataset_loaders():
    assert hasattr(RAGTruthLoader, "load")
    assert hasattr(HaluEvalLoader, "load")
