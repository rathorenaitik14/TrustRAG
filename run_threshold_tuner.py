"""
Threshold Tuner
================
Finds the optimal NLI thresholds on a validation split.
Tunes entailment_threshold and contradiction_threshold for NLIVerifier,
and relevance_threshold for CalibratedAbstentionVerifier.

Run:
    python run_threshold_tuner.py --dataset ragtruth --n_val 150 --output results/
"""
import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrustRAG.ThresholdTuner")


def tune_nli_thresholds(val_cases, verifier, ent_range=(0.3, 0.8), con_range=(0.3, 0.8), steps=10):
    """
    Grid-search over (entailment_threshold, contradiction_threshold).
    Returns best thresholds and macro-F1 on validation set.
    """
    import numpy as np
    from evaluation.stats import macro_f1

    best_f1 = -1.0
    best_thresholds = (0.5, 0.5)
    results = {}

    ent_values = np.linspace(*ent_range, steps).tolist()
    con_values = np.linspace(*con_range, steps).tolist()

    for ent in ent_values:
        for con in con_values:
            verifier.entailment_threshold = round(ent, 3)
            verifier.contradiction_threshold = round(con, 3)

            y_true, y_pred = [], []
            for case in val_cases:
                context = case.get("context", case.get("evidence_reference", ""))
                answer = case.get("injected_answer", case.get("answer", ""))
                true_label = case["ground_truth_label"]

                claim_evidence = {
                    "claim_id": case["case_id"],
                    "claim_text": answer,
                    "evidence": [{"evidence_id": "ev-1", "rank": 1,
                                  "text": context, "faiss_score": 1.0, "rerank_score": 1.0}]
                }
                try:
                    result = verifier.verify(claim_evidence)
                    pred = result.get("status", "UNSUPPORTED")
                except Exception:
                    pred = "UNSUPPORTED"

                y_true.append(true_label)
                y_pred.append(pred)

            f1 = macro_f1(y_true, y_pred)
            key = f"ent={ent:.2f}_con={con:.2f}"
            results[key] = round(f1, 4)

            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = (round(ent, 3), round(con, 3))

    return best_thresholds, best_f1, results


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="TrustRAG Threshold Tuner")
    parser.add_argument("--dataset", default="ragtruth",
                        choices=["ragtruth", "halueval", "all"])
    parser.add_argument("--n_val", type=int, default=150,
                        help="Number of validation cases (separate from test set)")
    parser.add_argument("--seed", type=int, default=123,
                        help="Different seed from test set (default 42)")
    parser.add_argument("--steps", type=int, default=8,
                        help="Grid steps per dimension (8x8=64 combos)")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    # Load validation data (different seed from test set)
    from run_public_eval import load_cases
    val_cases = load_cases(args.dataset, args.n_val, args.seed)
    logger.info(f"Validation set: {len(val_cases)} cases")

    results_summary = {}

    # ── Tune NLI (small) ─────────────────────────────────────────────────────
    logger.info("Tuning NLIVerifier (deberta-v3-small) ...")
    from verification.nli_verifier import NLIVerifier
    nli = NLIVerifier()
    best_nli, best_f1_nli, grid_nli = tune_nli_thresholds(
        val_cases, nli, steps=args.steps
    )
    logger.info(f"Best NLI thresholds: ent={best_nli[0]}, con={best_nli[1]} → F1={best_f1_nli:.4f}")
    results_summary["nli_verifier"] = {
        "best_entailment_threshold": best_nli[0],
        "best_contradiction_threshold": best_nli[1],
        "best_val_macro_f1": best_f1_nli,
    }

    # ── Tune MiniCheck ────────────────────────────────────────────────────────
    try:
        logger.info("Tuning MiniCheckVerifier ...")
        from verification.minicheck_verifier import MiniCheckVerifier
        mini = MiniCheckVerifier()
        best_mini, best_f1_mini, grid_mini = tune_nli_thresholds(
            val_cases, mini, steps=args.steps
        )
        logger.info(f"Best MiniCheck thresholds: ent={best_mini[0]}, con={best_mini[1]} → F1={best_f1_mini:.4f}")
        results_summary["minicheck_verifier"] = {
            "best_entailment_threshold": best_mini[0],
            "best_contradiction_threshold": best_mini[1],
            "best_val_macro_f1": best_f1_mini,
        }
    except Exception as e:
        logger.warning(f"MiniCheck tuning skipped: {e}")

    # ── Tune Abstention threshold ─────────────────────────────────────────────
    try:
        logger.info("Tuning abstention threshold ...")
        from verification.minicheck_verifier import MiniCheckVerifier
        from verification.abstention_verifier import CalibratedAbstentionVerifier

        base = MiniCheckVerifier(
            entailment_threshold=results_summary.get(
                "minicheck_verifier", {}).get("best_entailment_threshold", 0.5),
            contradiction_threshold=results_summary.get(
                "minicheck_verifier", {}).get("best_contradiction_threshold", 0.5),
        )
        abstaining = CalibratedAbstentionVerifier(base, relevance_threshold=0.30)

        # Build evidence results map for the tune_threshold method
        evidence_map = {}
        for case in val_cases:
            context = case.get("context", case.get("evidence_reference", ""))
            answer = case.get("injected_answer", case.get("answer", ""))
            evidence_map[case["case_id"]] = {
                "claim_id": case["case_id"],
                "claim_text": answer,
                "evidence": [{"evidence_id": "ev-1", "rank": 1,
                              "text": context, "faiss_score": 1.0, "rerank_score": 1.0}]
            }

        tune_result = abstaining.tune_threshold(val_cases, evidence_map)
        logger.info(
            f"Best abstention threshold: {tune_result['best_threshold']:.2f} "
            f"→ F1={tune_result['best_macro_f1']:.4f}"
        )
        results_summary["abstention"] = {
            "best_relevance_threshold": tune_result["best_threshold"],
            "best_val_macro_f1": tune_result["best_macro_f1"],
            "all_thresholds": tune_result["all_results"],
        }
    except Exception as e:
        logger.warning(f"Abstention tuning skipped: {e}")

    # ── Save results ─────────────────────────────────────────────────────────
    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.output, "tuned_thresholds.json")
    with open(out_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  TUNED THRESHOLDS")
    print("=" * 60)
    for name, vals in results_summary.items():
        print(f"\n  {name}:")
        for k, v in vals.items():
            if not isinstance(v, dict):
                print(f"    {k}: {v}")
    print(f"\n  Saved to: {out_path}")
    print("=" * 60)
    print("\n  → Copy these values into your verifier constructors")
    print("    before running run_public_eval.py for final results.\n")


if __name__ == "__main__":
    main()
