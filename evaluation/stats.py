"""
Statistical Evaluation Utilities
=================================
Publication-grade statistics: bootstrap confidence intervals,
McNemar's significance test, and per-class metrics with CIs.

All functions work on plain Python lists of string labels.
"""
import json
import math
import logging
import random
from typing import List, Tuple, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)

CLASSES = ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]


# ---------------------------------------------------------------------------
# Core metric functions (no external deps)
# ---------------------------------------------------------------------------

def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    """Compute macro-averaged F1 across SUPPORTED / UNSUPPORTED / CONTRADICTED."""
    f1s = []
    for cls in CLASSES:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def per_class_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, float]]:
    """Return precision, recall, F1 for each class."""
    result = {}
    for cls in CLASSES:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result[cls] = {"precision": prec, "recall": rec, "f1": f1}
    return result


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: List[str],
    y_pred: List[str],
    metric_fn: Callable[[List[str], List[str]], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)
    """
    rng = random.Random(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        bt = [y_true[i] for i in indices]
        bp = [y_pred[i] for i in indices]
        scores.append(metric_fn(bt, bp))

    scores.sort()
    lower_idx = int(math.floor((alpha / 2) * n_bootstrap))
    upper_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    lower_idx = max(0, lower_idx)
    upper_idx = min(n_bootstrap - 1, upper_idx)

    point = metric_fn(y_true, y_pred)
    return point, scores[lower_idx], scores[upper_idx]


def all_metrics_with_ci(
    y_true: List[str],
    y_pred: List[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute all metrics with 95% bootstrap CIs. Used in the final results table.

    Returns a dict ready for JSON serialisation.
    """
    n = len(y_true)
    if n == 0:
        return {"error": "No predictions provided."}

    point_f1, lo_f1, hi_f1 = bootstrap_ci(y_true, y_pred, macro_f1, n_bootstrap, seed=seed)
    point_acc, lo_acc, hi_acc = bootstrap_ci(y_true, y_pred, accuracy, n_bootstrap, seed=seed)

    per_class = per_class_metrics(y_true, y_pred)

    # Per-class CIs for F1
    for cls in CLASSES:
        def cls_f1(yt, yp, _cls=cls):
            tp = sum(t == _cls and p == _cls for t, p in zip(yt, yp))
            fp = sum(t != _cls and p == _cls for t, p in zip(yt, yp))
            fn = sum(t == _cls and p != _cls for t, p in zip(yt, yp))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        pt, lo, hi = bootstrap_ci(y_true, y_pred, cls_f1, n_bootstrap, seed=seed)
        per_class[cls]["f1_ci"] = (lo, hi)

    return {
        "n_samples": n,
        "macro_f1": round(point_f1, 4),
        "macro_f1_ci_95": (round(lo_f1, 4), round(hi_f1, 4)),
        "accuracy": round(point_acc, 4),
        "accuracy_ci_95": (round(lo_acc, 4), round(hi_acc, 4)),
        "per_class": {
            cls: {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in metrics.items()}
            for cls, metrics in per_class.items()
        },
        "label_distribution": {
            cls: y_true.count(cls) for cls in CLASSES
        }
    }


# ---------------------------------------------------------------------------
# McNemar's test (two-classifier significance test)
# ---------------------------------------------------------------------------

def mcnemar_test(
    y_true: List[str],
    y_pred_a: List[str],
    y_pred_b: List[str],
) -> Dict[str, float]:
    """
    McNemar's test comparing two classifiers A and B on the same test set.

    Returns p-value and chi-squared statistic.
    A p-value < 0.05 means B is significantly different from A.

    Uses the exact mid-p correction for small samples (n01 + n10 < 25).
    """
    # n01: A wrong, B right. n10: A right, B wrong.
    n01 = sum(
        a != t and b == t
        for t, a, b in zip(y_true, y_pred_a, y_pred_b)
    )
    n10 = sum(
        a == t and b != t
        for t, a, b in zip(y_true, y_pred_a, y_pred_b)
    )

    # Chi-squared with continuity correction
    total = n01 + n10
    if total == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n01": 0, "n10": 0,
                "significant_at_p05": False,
                "interpretation": "No discordant pairs — systems identical."}

    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)

    # Use scipy if available, otherwise approximate
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(1 - chi2_dist.cdf(chi2, df=1))
    except ImportError:
        # Rough approximation using normal distribution for large samples
        import math
        z = math.sqrt(chi2)
        p_value = 2 * (1 - _norm_cdf(abs(z)))

    sig = p_value < 0.05
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "n01": n01,
        "n10": n10,
        "significant_at_p05": sig,
        "interpretation": (
            f"TrustRAG is significantly {'better' if n01 > n10 else 'worse'} than baseline "
            f"(p={p_value:.4f})"
            if sig
            else f"No significant difference (p={p_value:.4f})"
        )
    }


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (no scipy needed)."""
    return (1 + math.erf(x / math.sqrt(2))) / 2


# ---------------------------------------------------------------------------
# Full results table builder (for paper)
# ---------------------------------------------------------------------------

def build_results_table(
    systems: Dict[str, Tuple[List[str], List[str]]],  # {name: (y_true, y_pred)}
    reference_system: str = "TrustRAG",
    n_bootstrap: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Build the paper's main results table.

    Parameters
    ----------
    systems          : Dict mapping system name → (y_true, y_pred)
    reference_system : Name of your system for significance testing
    n_bootstrap      : Bootstrap iterations

    Returns
    -------
    List of row dicts. Print with tabulate or export to CSV.
    """
    rows = []
    ref_pred = systems.get(reference_system, ([], []))[1]
    ref_true = systems.get(reference_system, ([], []))[0]

    for name, (y_true, y_pred) in systems.items():
        metrics = all_metrics_with_ci(y_true, y_pred, n_bootstrap=n_bootstrap)

        row: Dict[str, Any] = {
            "System": name,
            "n": metrics.get("n_samples", 0),
            "Macro-F1": f"{metrics['macro_f1']:.3f} [{metrics['macro_f1_ci_95'][0]:.3f}–{metrics['macro_f1_ci_95'][1]:.3f}]",
            "Acc": f"{metrics['accuracy']:.3f}",
        }

        # Per-class F1
        for cls in CLASSES:
            short = cls[:3]  # SUP, UNS, CON
            row[f"F1-{short}"] = round(metrics["per_class"][cls]["f1"], 3)

        # Significance vs reference
        if name != reference_system and ref_pred:
            sig = mcnemar_test(y_true, y_pred, ref_pred)
            row["vs TrustRAG (p)"] = sig["p_value"]
            row["Significant?"] = "✓" if sig["significant_at_p05"] else "✗"

        rows.append(row)

    return rows


def save_results(
    systems: Dict[str, Tuple[List[str], List[str]]],
    output_path: str,
    n_bootstrap: int = 1000,
):
    """
    Run all statistics and save a complete results JSON suitable for
    importing into a paper table.
    """
    output = {}
    for name, (y_true, y_pred) in systems.items():
        output[name] = all_metrics_with_ci(y_true, y_pred, n_bootstrap=n_bootstrap)

    # Add pairwise significance
    ref = "TrustRAG"
    if ref in systems:
        ref_true, ref_pred = systems[ref]
        output["significance_tests"] = {}
        for name, (y_true, y_pred) in systems.items():
            if name != ref:
                output["significance_tests"][f"{ref}_vs_{name}"] = mcnemar_test(
                    ref_true, ref_pred, y_pred
                )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    return output
