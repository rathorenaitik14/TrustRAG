"""
Full Public Benchmark Evaluation Runner
========================================
Evaluates TrustRAG and all baselines on RAGTruth + HaluEval.
Produces a results JSON file and a printable paper table.

Run:
    python run_public_eval.py --dataset ragtruth --n 500 --output results/
    python run_public_eval.py --dataset halueval --n 300 --output results/
    python run_public_eval.py --dataset all      --n 800 --output results/
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TrustRAG.Eval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env():
    """Load .env so GOOGLE_API_KEY is available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def print_table(rows: List[Dict[str, Any]]):
    """Pretty-print a list of dicts as an ASCII table."""
    if not rows:
        return
    headers = list(rows[0].keys())
    col_widths = {h: max(len(str(h)), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    sep = "+" + "+".join("-" * (col_widths[h] + 2) for h in headers) + "+"
    header_row = "|" + "|".join(f" {h:<{col_widths[h]}} " for h in headers) + "|"
    print(sep)
    print(header_row)
    print(sep)
    for row in rows:
        print("|" + "|".join(f" {str(row.get(h, '')):<{col_widths[h]}} " for h in headers) + "|")
    print(sep)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_cases(dataset: str, n: int, seed: int) -> List[dict]:
    """Load and combine benchmark cases."""
    from evaluation.public_datasets import RAGTruthLoader, HaluEvalLoader

    cases = []
    if dataset in ("ragtruth", "all"):
        logger.info("Loading RAGTruth ...")
        loader = RAGTruthLoader()
        n_rt = n if dataset == "ragtruth" else int(n * 0.625)  # 500/800
        cases.extend(loader.load(n_samples=n_rt, seed=seed))

    if dataset in ("halueval", "all"):
        logger.info("Loading HaluEval-QA ...")
        loader = HaluEvalLoader()
        n_he = n if dataset == "halueval" else int(n * 0.375)  # 300/800
        cases.extend(loader.load(n_samples=n_he, seed=seed))

    if not cases:
        raise ValueError(f"Unknown dataset '{dataset}'. Use: ragtruth | halueval | all")

    logger.info(f"Total cases loaded: {len(cases)}")
    return cases


# ---------------------------------------------------------------------------
# TrustRAG pipeline
# ---------------------------------------------------------------------------

def build_trustrag_pipeline(use_hybrid: bool = True, use_minicheck: bool = True,
                             use_abstention: bool = True, use_numeric: bool = True):
    """
    Build the TrustRAG pipeline from existing components.
    Returns a callable: verify_fn(question, context, answer) → dict
    """
    # Select verifier
    if use_minicheck:
        from verification.minicheck_verifier import MiniCheckVerifier
        base_verifier = MiniCheckVerifier()
    else:
        from verification.nli_verifier import NLIVerifier
        base_verifier = NLIVerifier()

    # Numeric checker
    if use_numeric:
        from verification.numeric_checker import NumericContradictionChecker
        numeric_checker = NumericContradictionChecker()
    else:
        numeric_checker = None

    # Abstention wrapper
    if use_abstention:
        from verification.abstention_verifier import CalibratedAbstentionVerifier, RelevanceScorer
        verifier_with_abstention = CalibratedAbstentionVerifier(
            base_verifier=base_verifier,
            relevance_threshold=0.30,  # tune later
        )
    else:
        verifier_with_abstention = None

    def verify_fn(question: str, context: str, answer: str) -> dict:
        """
        Simulate TrustRAG verification using the public dataset context directly
        (bypasses FAISS since we have the ground-truth passage).
        """
        # Build a mock ClaimEvidenceResult for the verifier
        claim_evidence = {
            "claim_id": "eval-claim",
            "claim_text": answer,
            "evidence": [
                {
                    "evidence_id": "eval-ev-1",
                    "rank": 1,
                    "text": context,
                    "faiss_score": 1.0,
                    "rerank_score": 1.0,
                }
            ]
        }

        # Run with abstention
        if verifier_with_abstention:
            result = verifier_with_abstention.verify(claim_evidence)
        else:
            result = base_verifier.verify(claim_evidence)

        # Optional: numeric override
        if numeric_checker and result.get("status") != "CONTRADICTED":
            single_pred = base_verifier.verify_single_evidence(answer, context)
            overridden = numeric_checker.override_nli_result(single_pred, answer, context)
            if overridden.get("status") == "CONTRADICTED":
                result["status"] = "CONTRADICTED"
                result["numeric_override"] = True

        return {"label": result["status"], "score": result.get("score", 0.0)}

    return verify_fn


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    cases: List[dict],
    system_name: str,
    verify_fn,
) -> Tuple[List[str], List[str], float]:
    """
    Run verify_fn over all cases. Returns (y_true, y_pred, avg_latency_ms).
    """
    y_true, y_pred = [], []
    latencies = []

    logger.info(f"Running evaluation for: {system_name} ({len(cases)} cases)")

    for i, case in enumerate(cases):
        question = case.get("question", "")
        context  = case.get("context", case.get("evidence_reference", ""))
        answer   = case.get("injected_answer", case.get("answer", ""))
        true_label = case["ground_truth_label"]

        t0 = time.perf_counter()
        try:
            result = verify_fn(question, context, answer)
            pred_label = result.get("label", "UNSUPPORTED")
        except Exception as e:
            logger.warning(f"Case {case['case_id']} failed: {e}. Defaulting to UNSUPPORTED.")
            pred_label = "UNSUPPORTED"
        latency_ms = (time.perf_counter() - t0) * 1000

        # Map UNVERIFIABLE → UNSUPPORTED for metric calculation
        # (it is not in the ground-truth label set)
        if pred_label == "UNVERIFIABLE":
            pred_label_for_metric = "UNSUPPORTED"
        else:
            pred_label_for_metric = pred_label

        y_true.append(true_label)
        y_pred.append(pred_label_for_metric)
        latencies.append(latency_ms)

        if (i + 1) % 50 == 0:
            from evaluation.stats import macro_f1, accuracy as acc_fn
            running_f1 = macro_f1(y_true, y_pred)
            logger.info(
                f"  [{i+1}/{len(cases)}] Running Macro-F1={running_f1:.4f} | "
                f"Avg latency={sum(latencies)/len(latencies):.1f}ms"
            )

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return y_true, y_pred, avg_latency


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    load_env()

    parser = argparse.ArgumentParser(description="TrustRAG Public Benchmark Evaluation")
    parser.add_argument("--dataset", default="ragtruth",
                        choices=["ragtruth", "halueval", "all"],
                        help="Which dataset(s) to evaluate on.")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of test cases.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results",
                        help="Directory to save results JSON.")
    parser.add_argument("--baselines", nargs="+",
                        default=["plain_nli", "llm_judge"],
                        choices=["plain_nli", "llm_judge", "ragas", "minicheck"],
                        help="Which baselines to run.")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--no-minicheck", action="store_true")
    parser.add_argument("--no-abstention", action="store_true")
    parser.add_argument("--no-numeric", action="store_true")
    args = parser.parse_args()

    # Load data
    cases = load_cases(args.dataset, args.n, args.seed)

    # Build systems dict: name → verify_fn
    systems_fns = {}

    # TrustRAG (full system)
    logger.info("Building TrustRAG pipeline ...")
    systems_fns["TrustRAG"] = build_trustrag_pipeline(
        use_hybrid=not args.no_hybrid,
        use_minicheck=not args.no_minicheck,
        use_abstention=not args.no_abstention,
        use_numeric=not args.no_numeric,
    )

    # Baselines
    from evaluation.baselines import (
        PlainNLIBaseline, LLMJudgeBaseline,
        RAGASFaithfulness, MiniCheckBaseline,
    )

    baseline_map = {
        "plain_nli":  ("Plain NLI (no retrieval)", lambda: PlainNLIBaseline()),
        "llm_judge":  ("LLM-as-Judge (Gemini)",    lambda: LLMJudgeBaseline()),
        "ragas":      ("RAGAS Faithfulness",        lambda: RAGASFaithfulness()),
        "minicheck":  ("MiniCheck (standalone)",    lambda: MiniCheckBaseline()),
    }

    for key in args.baselines:
        name, factory = baseline_map[key]
        logger.info(f"Loading baseline: {name}")
        obj = factory()
        systems_fns[name] = lambda q, c, a, _obj=obj: _obj.verify(q, c, a)

    # Run all systems
    all_results: Dict[str, Tuple[List[str], List[str]]] = {}
    latency_report = {}

    for sys_name, verify_fn in systems_fns.items():
        y_true, y_pred, avg_lat = run_evaluation(cases, sys_name, verify_fn)
        all_results[sys_name] = (y_true, y_pred)
        latency_report[sys_name] = avg_lat

    # Compute statistics
    logger.info("Computing statistics ...")
    from evaluation.stats import build_results_table, save_results

    table_rows = build_results_table(all_results, reference_system="TrustRAG")
    for row in table_rows:
        row["Latency (ms)"] = round(latency_report.get(row["System"], 0), 1)

    print("\n" + "=" * 80)
    print("  RESULTS TABLE")
    print("=" * 80)
    print_table(table_rows)

    # Save
    Path(args.output).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(args.output, f"results_{args.dataset}_{timestamp}.json")
    saved = save_results(all_results, out_file)

    # Also save latency
    saved["latency_ms"] = latency_report
    saved["args"] = vars(args)
    saved["n_cases"] = len(cases)
    with open(out_file, "w") as f:
        json.dump(saved, f, indent=2)

    logger.info(f"Results saved to {out_file}")
    print(f"\n✓ Results saved to {out_file}")


if __name__ == "__main__":
    main()
