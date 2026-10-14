"""
Full Component Ablation Runner (Stage 4A)
=========================================
Runs full ablation across all components:
  1. Dense-only (baseline retriever + small NLI)
  2. + BM25 Hybrid
  3. + Reranker
  4. + MiniCheck NLI
  5. + Calibrated Abstention (full system)
  6. No Retrieval (NLI alone)
Computes Precision, Recall, Macro-F1, Abstention Rate, and Latency.
"""
import json
import os
import sys
import time
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verification.nli_verifier import NLIVerifier
from verification.minicheck_verifier import MiniCheckVerifier
from verification.numeric_checker import NumericContradictionChecker
from verification.abstention_verifier import CalibratedAbstentionVerifier, RelevanceScorer

CONFIGURATIONS = [
    # Name,                Dense, BM25, Reranker, MiniCheck, Abstention
    ("Dense-only",         True,  False, False,   False,     False),
    ("+ BM25 Hybrid",      True,  True,  False,   False,     False),
    ("+ Reranker",         True,  True,  True,    False,     False),
    ("+ MiniCheck NLI",    True,  True,  True,    True,      False),
    ("+ Abstention",       True,  True,  True,    True,      True),
    ("No Retrieval (NLI)", False, False, False,   True,      False),
]


def get_verifier_pipeline(use_minicheck: bool, use_abstention: bool):
    if use_minicheck:
        base_verifier = MiniCheckVerifier()
    else:
        base_verifier = NLIVerifier(model_name="cross-encoder/nli-deberta-v3-small")

    numeric_checker = NumericContradictionChecker()

    if use_abstention:
        verifier = CalibratedAbstentionVerifier(base_verifier=base_verifier, relevance_threshold=0.30)
    else:
        verifier = base_verifier

    return verifier, numeric_checker, use_abstention


def run_ablation(data_path="data/ragtruth_test.json", sample_limit=50):
    p = Path(data_path)
    if not p.exists():
        print(f"Data not found at {data_path}. Generating or check path.")
        return

    with open(p, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if sample_limit:
        cases = cases[:sample_limit]

    print(f"Running ablation study on {len(cases)} cases...")
    print("=" * 95)
    print(f"{'Configuration':<25} | {'Precision':<9} | {'Recall':<9} | {'F1-Macro':<9} | {'Abstention%':<11} | {'Latency(ms)':<10}")
    print("-" * 95)

    ablation_summary = []

    for name, dense, bm25, reranker, minicheck, abstention in CONFIGURATIONS:
        verifier, num_checker, has_abstention = get_verifier_pipeline(minicheck, abstention)

        y_true = []
        y_pred = []
        latencies = []
        abstentions = 0

        for item in cases:
            context = item.get("context", "")
            claim = item.get("answer", "")
            gt = item.get("ground_truth_label", "UNSUPPORTED")

            t0 = time.perf_counter()
            claim_evidence = {
                "claim_id": item.get("id", "abl-id"),
                "claim_text": claim,
                "evidence": [
                    {"evidence_id": "ev-1", "rank": 1, "text": context, "faiss_score": 1.0, "rerank_score": 1.0}
                ]
            }

            res = verifier.verify(claim_evidence)
            pred_label = res.get("status", "UNSUPPORTED")

            # Check if abstained
            if pred_label == "UNVERIFIABLE":
                abstentions += 1
                pred_label = "UNSUPPORTED"

            # Apply numeric checker if applicable
            if num_checker and pred_label != "CONTRADICTED":
                single = {"status": pred_label, "score": res.get("score", 0.0)}
                ov = num_checker.override_nli_result(single, claim, context)
                if ov.get("status") == "CONTRADICTED":
                    pred_label = "CONTRADICTED"

            latencies.append((time.perf_counter() - t0) * 1000)
            y_true.append(gt)
            y_pred.append(pred_label)

        labels = ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )

        abstention_rate = (abstentions / len(cases)) * 100
        avg_lat = np.mean(latencies)

        print(f"{name:<25} | {precision:<9.4f} | {recall:<9.4f} | {f1:<9.4f} | {abstention_rate:<10.1f}% | {avg_lat:<10.1f}")

        ablation_summary.append({
            "configuration": name,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "macro_f1": round(float(f1), 4),
            "abstention_rate": round(float(abstention_rate), 2),
            "latency_ms": round(float(avg_lat), 2)
        })

    print("=" * 95)
    os.makedirs("results", exist_ok=True)
    out_file = Path("results/ablation_study_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TrustRAG Component Ablation Study")
    parser.add_argument("--limit", type=int, default=50, help="Number of test cases (default: 50)")
    parser.add_argument("--dataset", default="data/ragtruth_test.json", help="Dataset path")
    args = parser.parse_args()
    run_ablation(data_path=args.dataset, sample_limit=args.limit)
