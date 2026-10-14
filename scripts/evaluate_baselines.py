import json
import time
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.baselines import PlainNLIBaseline, LLMJudgeBaseline, MiniCheckBaseline, RAGASFaithfulness
from evaluation.stats import bootstrap_ci, mcnemar_test, macro_f1
from verification.nli_verifier import NLIVerifier
from verification.minicheck_verifier import MiniCheckVerifier
from verification.numeric_checker import NumericContradictionChecker
from verification.abstention_verifier import CalibratedAbstentionVerifier


class TrustRAGWrapper:
    """Wraps TrustRAG pipeline for unified baseline evaluation."""
    def __init__(self, use_minicheck: bool = True, use_abstention: bool = True):
        if use_minicheck:
            base_v = MiniCheckVerifier()
        else:
            base_v = NLIVerifier(model_name="cross-encoder/nli-deberta-v3-small")
        self.num_checker = NumericContradictionChecker()
        if use_abstention:
            self.verifier = CalibratedAbstentionVerifier(base_verifier=base_v, relevance_threshold=0.30)
        else:
            self.verifier = base_v

    def verify(self, question: str, context: str, answer: str) -> dict:
        claim_ev = {
            "claim_id": "eval-id",
            "claim_text": answer,
            "evidence": [{"evidence_id": "ev-1", "rank": 1, "text": context, "faiss_score": 1.0, "rerank_score": 1.0}]
        }
        res = self.verifier.verify(claim_ev)
        pred_label = res.get("status", "UNSUPPORTED")
        if pred_label == "UNVERIFIABLE":
            pred_label = "UNSUPPORTED"
        if self.num_checker and pred_label != "CONTRADICTED":
            ov = self.num_checker.override_nli_result({"status": pred_label, "score": res.get("score", 0.0)}, answer, context)
            if ov.get("status") == "CONTRADICTED":
                pred_label = "CONTRADICTED"
        return {"label": pred_label, "score": res.get("score", 0.0)}


def evaluate_system(system, data, name="System"):
    print(f"\nEvaluating {name} on {len(data)} cases...")
    y_true = []
    y_pred = []
    latencies = []

    for item in tqdm(data):
        context = item.get("context", "")
        claim = item.get("answer", "")
        gt = item.get("ground_truth_label", "UNSUPPORTED")

        start = time.perf_counter()
        try:
            res = system.verify(question=item.get("question", ""), context=context, answer=claim)
        except Exception as e:
            res = {"label": "UNSUPPORTED", "score": 0.0}
        end = time.perf_counter()

        latencies.append(end - start)
        y_true.append(gt)
        y_pred.append(res["label"])

    labels = ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_c, r_c, f_c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # Compute 95% Bootstrap CI for Macro F1
    mean_f1, ci_lower, ci_upper = bootstrap_ci(y_true, y_pred, macro_f1, n_bootstrap=500)

    print(f"--- Results for {name} ---")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"Mean Latency:    {np.mean(latencies)*1000:.2f} ms")

    return {
        "name": name,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f1_ci_95": [float(ci_lower), float(ci_upper)],
        "latency_ms": float(np.mean(latencies)*1000),
        "per_class": dict(zip(labels, [float(x) for x in f_c])),
        "y_true": y_true,
        "y_pred": y_pred
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baselines & TrustRAG")
    parser.add_argument("--limit", type=int, default=50, help="Number of cases to evaluate (default: 50)")
    parser.add_argument("--dataset", default="data/ragtruth_test.json", help="Path to evaluation JSON")
    parser.add_argument("--include_llm", action="store_true", help="Include Gemini LLM-as-judge")
    args = parser.parse_args()

    data_path = Path(args.dataset)
    if not data_path.exists():
        print(f"Data not found at {data_path}. Run load_ragtruth.py or load_halueval.py first.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.limit and args.limit > 0:
        data = data[:args.limit]
    print(f"Loaded {len(data)} test cases from {data_path}.")

    systems = [
        ("Plain NLI (no retrieval)", PlainNLIBaseline()),
        ("MiniCheck NLI (standalone)", MiniCheckBaseline(model_size="medium")),
        ("TrustRAG (DeBERTa-small)", TrustRAGWrapper(use_minicheck=False, use_abstention=True)),
        ("TrustRAG + MiniCheck (Ours)", TrustRAGWrapper(use_minicheck=True, use_abstention=True)),
    ]

    has_api_key = bool(os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if args.include_llm and has_api_key:
        systems.insert(1, ("LLM-as-Judge (Gemini)", LLMJudgeBaseline()))

    results = []
    for name, s in systems:
        res = evaluate_system(s, data, name=name)
        results.append(res)

    # Compute McNemar test comparing best TrustRAG against baselines
    ours = next((r for r in results if r["name"] == "TrustRAG + MiniCheck (Ours)"), results[-1])
    mcnemar_stats = {}
    for r in results:
        if r["name"] != ours["name"]:
            mc_res = mcnemar_test(r["y_true"], r["y_pred"], ours["y_pred"])
            p_val = float(mc_res.get("p_value", 1.0)) if isinstance(mc_res, dict) else float(mc_res)
            mcnemar_stats[f"{ours['name']} vs {r['name']}"] = p_val

    print("\n" + "=" * 95)
    print("                      HEAD-TO-HEAD PUBLICATION BENCHMARK TABLE")
    print("=" * 95)
    print(f"{'System':<32} | {'Precision':<9} | {'Recall':<9} | {'F1-Macro':<9} | {'95% CI':<16} | {'Latency(ms)':<10}")
    print("-" * 95)
    for r in results:
        ci_str = f"[{r['f1_ci_95'][0]:.3f}, {r['f1_ci_95'][1]:.3f}]"
        print(f"{r['name']:<32} | {r['precision']:<9.4f} | {r['recall']:<9.4f} | {r['f1']:<9.4f} | {ci_str:<16} | {r['latency_ms']:<10.2f}")
    print("=" * 95)

    print("\n--- Significance Tests (McNemar's p-value vs Ours) ---")
    for comp, p in mcnemar_stats.items():
        sig = "*** (p < 0.001)" if p < 0.001 else "** (p < 0.01)" if p < 0.01 else "* (p < 0.05)" if p < 0.05 else "n.s."
        print(f"  {comp}: p = {p:.5f} {sig}")

    os.makedirs("results", exist_ok=True)
    out_payload = {
        "dataset": str(data_path),
        "n_samples": len(data),
        "results": [{k: v for k, v in r.items() if k not in ("y_true", "y_pred")} for r in results],
        "mcnemar_significance": mcnemar_stats
    }
    with open("results/baseline_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print("\nSaved full evaluation results to results/baseline_eval_results.json")


if __name__ == "__main__":
    main()
