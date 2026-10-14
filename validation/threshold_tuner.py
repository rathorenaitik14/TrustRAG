"""
Validation Threshold Tuner (Stage 2B)
=====================================
Tunes NLI entailment and contradiction thresholds on a held-out validation set.
Guarantees test set integrity by optimizing hyper-parameters strictly on val_cases.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from evaluation.stats import macro_f1


def tune_thresholds(
    val_cases: List[Dict[str, Any]],
    verifier,
    entailment_range: Tuple[float, float] = (0.3, 0.8),
    contradiction_range: Tuple[float, float] = (0.3, 0.8),
    steps: int = 15
) -> Tuple[Tuple[float, float], float, Dict[str, float]]:
    """
    Grid-search over (entailment_threshold, contradiction_threshold) on validation split.

    Parameters
    ----------
    val_cases          : List of benchmark case dicts (held-out from test set)
    verifier           : Verifier object with .entailment_threshold and .contradiction_threshold
    entailment_range   : Min and max entailment threshold to test
    contradiction_range: Min and max contradiction threshold to test
    steps              : Number of discrete steps per dimension

    Returns
    -------
    (best_thresholds, best_f1, all_results_dict)
    """
    best_f1 = -1.0
    best_thresholds = (0.5, 0.5)
    results = {}

    ent_values = np.linspace(entailment_range[0], entailment_range[1], steps).tolist()
    con_values = np.linspace(contradiction_range[0], contradiction_range[1], steps).tolist()

    for ent_thresh in ent_values:
        for con_thresh in con_values:
            verifier.entailment_threshold = round(ent_thresh, 3)
            verifier.contradiction_threshold = round(con_thresh, 3)

            y_true = []
            y_pred = []

            for case in val_cases:
                context = case.get("context", case.get("evidence_reference", ""))
                answer = case.get("injected_answer", case.get("answer", ""))
                true_label = case["ground_truth_label"]

                claim_evidence = {
                    "claim_id": case.get("case_id", case.get("id", "val-claim")),
                    "claim_text": answer,
                    "evidence": [
                        {
                            "evidence_id": "val-ev-1",
                            "rank": 1,
                            "text": context,
                            "faiss_score": 1.0,
                            "rerank_score": 1.0
                        }
                    ]
                }

                try:
                    res = verifier.verify(claim_evidence)
                    pred = res.get("status", "UNSUPPORTED")
                except Exception:
                    pred = "UNSUPPORTED"

                y_true.append(true_label)
                y_pred.append(pred)

            f1 = macro_f1(y_true, y_pred)
            key = f"ent={ent_thresh:.2f}_con={con_thresh:.2f}"
            results[key] = round(f1, 4)

            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = (round(ent_thresh, 3), round(con_thresh, 3))

    return best_thresholds, best_f1, results
