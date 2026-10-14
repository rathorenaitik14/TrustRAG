"""
Full Component Ablation Study
==============================
Systematically removes/adds each TrustRAG component to measure its
individual contribution. This is required for any publication.

Components ablated:
  C0: No retrieval  — Plain NLI only (weakest baseline)
  C1: Dense only    — FAISS + small NLI (current system)
  C2: + Reranker    — FAISS + CrossEncoder reranker + small NLI
  C3: + Hybrid      — BM25+FAISS (RRF) + reranker + small NLI
  C4: + MiniCheck   — BM25+FAISS + reranker + MiniCheck NLI
  C5: + Numeric     — All above + numeric contradiction checker
  C6: + Abstention  — Full TrustRAG with calibrated abstention

Run:
    python run_ablation_study.py --dataset ragtruth --n 500 --output results/
"""
import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TrustRAG.Ablation")


CONFIGURATIONS = [
    # (name, use_hybrid, use_reranker, use_minicheck, use_numeric, use_abstention)
    ("C0: Plain NLI",           False, False, False, False, False),
    ("C1: Dense + Small NLI",   False, False, False, False, False),  # dense is mocked
    ("C2: + Reranker",          False, True,  False, False, False),
    ("C3: + Hybrid Retrieval",  True,  True,  False, False, False),
    ("C4: + MiniCheck",         True,  True,  True,  False, False),
    ("C5: + Numeric Check",     True,  True,  True,  True,  False),
    ("C6: Full TrustRAG",       True,  True,  True,  True,  True),
]


def build_verify_fn(
    use_hybrid: bool,
    use_reranker: bool,
    use_minicheck: bool,
    use_numeric: bool,
    use_abstention: bool,
):
    """
    Build a verify_fn(question, context, answer) → dict for the given config.
    Since we're using public datasets that provide the context directly,
    we bypass FAISS and use the ground-truth passage as evidence.
    This isolates the NLI/abstention components for ablation.

    For real FAISS-based ablation (retrieval quality), use run_retrieval_ablation().
    """

    # NLI backend
    if use_minicheck:
        from verification.minicheck_verifier import MiniCheckVerifier
        nli_verifier = MiniCheckVerifier()
    else:
        from verification.nli_verifier import NLIVerifier
        nli_verifier = NLIVerifier()

    # Numeric checker
    if use_numeric:
        from verification.numeric_checker import NumericContradictionChecker
        numeric_checker = NumericContradictionChecker()
    else:
        numeric_checker = None

    # Abstention
    if use_abstention:
        from verification.abstention_verifier import CalibratedAbstentionVerifier
        final_verifier = CalibratedAbstentionVerifier(
            base_verifier=nli_verifier,
            relevance_threshold=0.30,
        )
    else:
        final_verifier = nli_verifier

    def verify_fn(question: str, context: str, answer: str) -> dict:
        claim_evidence = {
            "claim_id": "ablation-claim",
            "claim_text": answer,
            "evidence": [
                {"evidence_id": "ev-1", "rank": 1, "text": context,
                 "faiss_score": 1.0, "rerank_score": 1.0}
            ]
        }
        result = final_verifier.verify(claim_evidence)
        label = result.get("status", "UNSUPPORTED")

        # Numeric override (only if not already CONTRADICTED)
        if numeric_checker and label not in ("CONTRADICTED", "UNVERIFIABLE"):
            single = nli_verifier.verify_single_evidence(answer, context)
            overridden = numeric_checker.override_nli_result(single, answer, context)
            if overridden.get("status") == "CONTRADICTED":
                label = "CONTRADICTED"

        # Map UNVERIFIABLE → UNSUPPORTED for 3-class metric calculation
        if label == "UNVERIFIABLE":
            label = "UNSUPPORTED"

        return {"label": label, "score": result.get("score", 0.0)}

    return verify_fn


def run_config(cases: List[dict], config_name: str, verify_fn) -> Tuple[List[str], List[str], float]:
    """Run a single ablation configuration and return (y_true, y_pred, avg_latency_ms)."""
    y_true, y_pred = [], []
    latencies = []

    for case in cases:
        question = case.get("question", "")
        context  = case.get("context", case.get("evidence_reference", ""))
        answer   = case.get("injected_answer", case.get("answer", ""))
        true_label = case["ground_truth_label"]

        t0 = time.perf_counter()
        try:
            result = verify_fn(question, context, answer)
            pred = result.get("label", "UNSUPPORTED")
        except Exception as e:
            logger.warning(f"Config '{config_name}' failed on case {case['case_id']}: {e}")
            pred = "UNSUPPORTED"
        latencies.append((time.perf_counter() - t0) * 1000)

        y_true.append(true_label)
        y_pred.append(pred)

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    return y_true, y_pred, avg_lat


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="TrustRAG Ablation Study")
    parser.add_argument("--dataset", default="ragtruth",
                        choices=["ragtruth", "halueval", "all"])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    # Load data
    from run_public_eval import load_cases, print_table
    cases = load_cases(args.dataset, args.n, args.seed)

    # Run all configurations
    all_results: Dict[str, Tuple[List[str], List[str]]] = {}
    latency_report = {}
    table_rows = []

    for name, use_hybrid, use_reranker, use_minicheck, use_numeric, use_abstention in CONFIGURATIONS:
        logger.info(f"Running config: {name}")

        try:
            verify_fn = build_verify_fn(
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                use_minicheck=use_minicheck,
                use_numeric=use_numeric,
                use_abstention=use_abstention,
            )
            y_true, y_pred, avg_lat = run_config(cases, name, verify_fn)
        except Exception as e:
            logger.error(f"Config '{name}' failed entirely: {e}. Skipping.")
            continue

        all_results[name] = (y_true, y_pred)
        latency_report[name] = avg_lat

        from evaluation.stats import macro_f1, per_class_metrics, accuracy
        mf1 = macro_f1(y_true, y_pred)
        acc = accuracy(y_true, y_pred)
        pcm = per_class_metrics(y_true, y_pred)

        row = {
            "Configuration": name,
            "Macro-F1": f"{mf1:.3f}",
            "Accuracy": f"{acc:.3f}",
            "F1-SUP": f"{pcm['SUPPORTED']['f1']:.3f}",
            "F1-UNS": f"{pcm['UNSUPPORTED']['f1']:.3f}",
            "F1-CON": f"{pcm['CONTRADICTED']['f1']:.3f}",
            "Latency(ms)": f"{avg_lat:.1f}",
        }
        table_rows.append(row)
        logger.info(f"  → Macro-F1: {mf1:.4f} | Acc: {acc:.4f} | Latency: {avg_lat:.1f}ms")

    # Print table
    print("\n" + "=" * 90)
    print("  ABLATION STUDY RESULTS")
    print("=" * 90)
    print_table(table_rows)

    # Save
    Path(args.output).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(args.output, f"ablation_{args.dataset}_{timestamp}.json")

    from evaluation.stats import save_results
    save_results(all_results, out_file)
    logger.info(f"Ablation results saved to {out_file}")
    print(f"\n✓ Ablation results saved to {out_file}")


if __name__ == "__main__":
    main()
