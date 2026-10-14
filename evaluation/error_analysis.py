"""
Error Analysis Module
======================
Categorises verification failures into interpretable buckets for the
error analysis section of the paper. Also generates an HTML report.

Error categories (for Section 5 of the paper):
  1. NUMERICAL     – mismatched numbers that NLI missed
  2. PARAPHRASE    – semantically equivalent text, NLI confused by wording
  3. NEGATION      – logical negation the model missed (e.g., "no role" vs "essential role")
  4. RETRIEVAL_FAIL– correct claim, but wrong chunk was retrieved
  5. MULTI_HOP     – claim requires reasoning across 2+ passages
  6. OTHER         – unclassified failure

Usage:
    analyzer = ErrorAnalyzer()
    report = analyzer.analyze(cases, y_true, y_pred, evidence_texts)
    analyzer.save_html(report, "results/error_analysis.html")
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic category detectors
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?\s*(?:%|ZB|GB|TB|MB|billion|million|thousand)?\b',
                         re.IGNORECASE)
_NEGATION_RE = re.compile(
    r'\b(no|not|never|none|cannot|can\'t|impossible|without|zero|0)\b', re.IGNORECASE
)


def _has_numbers(text: str) -> bool:
    return bool(_NUMBER_RE.search(text))


def _has_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(text))


def _cosine_sim(a: str, b: str) -> float:
    """Quick token-overlap-based similarity (no model needed)."""
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def classify_error(
    claim: str,
    evidence: str,
    true_label: str,
    pred_label: str,
    retrieved_correctly: Optional[bool] = None,
) -> str:
    """
    Classify a single prediction error into a category.

    Parameters
    ----------
    claim             : The claim text being verified.
    evidence          : The top-1 retrieved evidence text.
    true_label        : Ground truth label.
    pred_label        : Predicted label.
    retrieved_correctly : None = unknown. True/False = whether the right evidence was retrieved.
    """
    if true_label == pred_label:
        return "CORRECT"

    # Retrieval failure: evidence not relevant (low overlap with claim)
    ev_sim = _cosine_sim(claim, evidence)
    if retrieved_correctly is False or (retrieved_correctly is None and ev_sim < 0.1):
        return "RETRIEVAL_FAIL"

    # Numerical: claim or evidence contains numbers
    if _has_numbers(claim) or _has_numbers(evidence):
        return "NUMERICAL"

    # Negation: claim or true label involves logical negation
    if _has_negation(claim) or (true_label == "CONTRADICTED" and _has_negation(evidence)):
        return "NEGATION"

    # Paraphrase: high overlap but still wrong (NLI confused by phrasing)
    if ev_sim > 0.4:
        return "PARAPHRASE"

    return "OTHER"


# ---------------------------------------------------------------------------
# Error Analyzer
# ---------------------------------------------------------------------------

class ErrorAnalyzer:
    """
    Performs systematic error analysis on a set of predictions.
    """

    CATEGORIES = ["CORRECT", "NUMERICAL", "NEGATION", "PARAPHRASE",
                  "RETRIEVAL_FAIL", "MULTI_HOP", "OTHER"]

    def analyze(
        self,
        cases: List[Dict],
        y_true: List[str],
        y_pred: List[str],
        evidence_texts: Optional[List[str]] = None,
        max_examples: int = 50,
    ) -> Dict[str, Any]:
        """
        Analyse errors and return a structured report.

        Parameters
        ----------
        cases          : List of UnifiedBenchmarkCase dicts.
        y_true         : Ground truth labels.
        y_pred         : Predicted labels.
        evidence_texts : Top-1 retrieved evidence for each case (optional).
        max_examples   : Max failure examples to include in report.

        Returns
        -------
        Report dict with category counts and examples.
        """
        if evidence_texts is None:
            evidence_texts = [
                c.get("context", c.get("evidence_reference", "")) for c in cases
            ]

        category_counts: Dict[str, int] = {cat: 0 for cat in self.CATEGORIES}
        category_examples: Dict[str, List[Dict]] = {cat: [] for cat in self.CATEGORIES}

        for i, (case, true, pred, ev) in enumerate(
            zip(cases, y_true, y_pred, evidence_texts)
        ):
            claim = case.get("injected_answer", case.get("answer", ""))
            category = classify_error(claim, ev, true, pred)
            category_counts[category] += 1

            if category != "CORRECT" and len(category_examples[category]) < max_examples:
                category_examples[category].append({
                    "case_id": case.get("case_id", f"case-{i}"),
                    "question": case.get("question", "")[:120],
                    "claim": claim[:200],
                    "evidence": ev[:200],
                    "true_label": true,
                    "pred_label": pred,
                    "category": category,
                })

        total = len(cases)
        errors = total - category_counts["CORRECT"]
        accuracy = category_counts["CORRECT"] / total if total > 0 else 0.0

        return {
            "total_cases": total,
            "correct": category_counts["CORRECT"],
            "errors": errors,
            "accuracy": round(accuracy, 4),
            "category_counts": category_counts,
            "category_percentages": {
                cat: round(cnt / max(errors, 1) * 100, 1)
                for cat, cnt in category_counts.items()
                if cat != "CORRECT"
            },
            "examples": category_examples,
        }

    def print_report(self, report: Dict[str, Any]):
        """Print a formatted error analysis table to stdout."""
        print("\n" + "=" * 60)
        print(f"  ERROR ANALYSIS  (n={report['total_cases']}, acc={report['accuracy']:.3f})")
        print("=" * 60)
        print(f"  Correct: {report['correct']} / Total errors: {report['errors']}")
        print("")
        print(f"  {'Category':<20} {'Count':>6} {'% of Errors':>12}")
        print(f"  {'-'*20} {'-'*6} {'-'*12}")

        for cat, count in report["category_counts"].items():
            if cat == "CORRECT":
                continue
            pct = report["category_percentages"].get(cat, 0.0)
            print(f"  {cat:<20} {count:>6} {pct:>11.1f}%")

        print("=" * 60)

        # Print 2 examples per non-trivial category
        print("\n  Sample Failure Cases:")
        print("  " + "-" * 58)
        for cat, examples in report["examples"].items():
            if not examples or cat == "CORRECT":
                continue
            print(f"\n  [{cat}] (showing {min(2, len(examples))}/{len(examples)} examples)")
            for ex in examples[:2]:
                print(f"    ID      : {ex['case_id']}")
                print(f"    Claim   : {ex['claim'][:100]}...")
                print(f"    Evidence: {ex['evidence'][:100]}...")
                print(f"    True    : {ex['true_label']}  → Predicted: {ex['pred_label']}")
                print()

    def save_html(self, report: Dict[str, Any], output_path: str):
        """Save error analysis as a self-contained HTML file for the appendix."""
        rows = ""
        for cat, examples in report["examples"].items():
            if cat == "CORRECT":
                continue
            for ex in examples:
                rows += f"""
                <tr class="cat-{cat.lower()}">
                    <td>{ex['case_id']}</td>
                    <td><code>{cat}</code></td>
                    <td>{ex['true_label']}</td>
                    <td>{ex['pred_label']}</td>
                    <td>{ex['claim'][:150]}</td>
                    <td>{ex['evidence'][:150]}</td>
                </tr>"""

        summary_rows = ""
        for cat, count in report["category_counts"].items():
            if cat == "CORRECT":
                continue
            pct = report["category_percentages"].get(cat, 0)
            summary_rows += f"<tr><td>{cat}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TrustRAG Error Analysis</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2em; }}
  h1 {{ color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 0.85em; }}
  th {{ background: #2c3e50; color: white; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .cat-numerical {{ background: #fff3cd; }}
  .cat-negation {{ background: #f8d7da; }}
  .cat-paraphrase {{ background: #d1ecf1; }}
  .cat-retrieval_fail {{ background: #d4edda; }}
  .cat-other {{ background: #e2e3e5; }}
  .stat {{ font-size: 1.1em; margin: 0.5em 0; }}
</style>
</head>
<body>
<h1>TrustRAG — Error Analysis Report</h1>
<p class="stat">Total cases: <strong>{report['total_cases']}</strong> |
   Correct: <strong>{report['correct']}</strong> |
   Accuracy: <strong>{report['accuracy']:.3f}</strong></p>

<h2>Error Category Summary</h2>
<table>
  <tr><th>Category</th><th>Count</th><th>% of Errors</th></tr>
  {summary_rows}
</table>

<h2>Failure Examples</h2>
<table>
  <tr>
    <th>Case ID</th><th>Category</th><th>True</th><th>Pred</th>
    <th>Claim (truncated)</th><th>Evidence (truncated)</th>
  </tr>
  {rows}
</table>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Error analysis HTML saved to {output_path}")


def main():
    import argparse
    import json
    import os
    import sys
    from pathlib import Path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from verification.nli_verifier import NLIVerifier

    parser = argparse.ArgumentParser(description="TrustRAG Error Analysis")
    parser.add_argument("--dataset", default="data/ragtruth_test.json")
    parser.add_argument("--limit", type=int, default=50, help="Number of test cases to analyze (min 50)")
    parser.add_argument("--output_html", default="results/error_analysis.html")
    parser.add_argument("--output_json", default="results/error_analysis.json")
    args = parser.parse_args()

    data_path = Path(args.dataset)
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        cases = json.load(f)[:args.limit]

    print(f"Running error analysis on {len(cases)} cases...")
    verifier = NLIVerifier(model_name="cross-encoder/nli-deberta-v3-small")

    y_true, y_pred, ev_texts = [], [], []
    for c in cases:
        claim = c.get("answer", "")
        ev = c.get("context", "")
        gt = c.get("ground_truth_label", "UNSUPPORTED")

        res = verifier.verify_single_evidence(claim, ev)
        pred = res.get("status", "UNSUPPORTED")

        y_true.append(gt)
        y_pred.append(pred)
        ev_texts.append(ev)

    analyzer = ErrorAnalyzer()
    report = analyzer.analyze(cases, y_true, y_pred, ev_texts)

    print("\n" + "=" * 70)
    print("                     ERROR ANALYSIS BREAKDOWN")
    print("=" * 70)
    print(f"{'Error Category':<25} | {'Count':<8} | {'% of Failures':<15}")
    print("-" * 70)
    for cat, count in report["category_counts"].items():
        if cat == "CORRECT":
            continue
        pct = report["category_percentages"].get(cat, 0.0)
        print(f"{cat:<25} | {count:<8} | {pct:<14.1f}%")
    print("=" * 70)
    print(f"Total Cases: {report['total_cases']} | Correct: {report['correct']} | Accuracy: {report['accuracy']:.3f}\n")

    os.makedirs("results", exist_ok=True)
    analyzer.save_html(report, args.output_html)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved HTML report to {args.output_html}")
    print(f"Saved JSON report to {args.output_json}")


if __name__ == "__main__":
    main()

