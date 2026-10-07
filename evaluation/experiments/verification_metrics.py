from typing import List, Dict, Any
from .models import VerificationMetricsResult

class VerificationMetricsCalculator:
    def calculate(self, predictions: List[Dict[str, str]]) -> VerificationMetricsResult:
        """
        predictions: [{'true_label': '...', 'pred_label': '...'}, ...]
        """
        classes = ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
        stats = {c: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for c in classes}
        
        confusion = {r: {c: 0 for c in classes} for r in classes}
        
        correct = 0
        total = len(predictions)
        
        # Binary problematic mapping:
        # PROBLEMATIC = UNSUPPORTED or CONTRADICTED
        # NON-PROBLEMATIC = SUPPORTED
        bin_tp, bin_fp, bin_fn, bin_tn = 0, 0, 0, 0
        
        for p in predictions:
            t = p.get("true_label", "")
            pr = p.get("pred_label", "")
            
            if t in classes and pr in classes:
                confusion[t][pr] += 1
                
            if t == pr:
                correct += 1
                
            for c in classes:
                if t == c and pr == c: stats[c]["TP"] += 1
                elif t != c and pr == c: stats[c]["FP"] += 1
                elif t == c and pr != c: stats[c]["FN"] += 1
                else: stats[c]["TN"] += 1
                
            # Binary
            t_prob = t in ["UNSUPPORTED", "CONTRADICTED"]
            pr_prob = pr in ["UNSUPPORTED", "CONTRADICTED"]
            
            if t_prob and pr_prob: bin_tp += 1
            elif not t_prob and pr_prob: bin_fp += 1
            elif t_prob and not pr_prob: bin_fn += 1
            else: bin_tn += 1
            
        res = {}
        res["accuracy"] = (correct / total) if total > 0 else 0.0
        
        for c in classes:
            tp, fp, fn = stats[c]["TP"], stats[c]["FP"], stats[c]["FN"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            
            label = c.lower()
            res[f"precision_{label}"] = prec
            res[f"recall_{label}"] = rec
            res[f"f1_{label}"] = f1
            
        res["macro_precision"] = sum(res[f"precision_{c.lower()}"] for c in classes) / 3
        res["macro_recall"] = sum(res[f"recall_{c.lower()}"] for c in classes) / 3
        res["macro_f1"] = sum(res[f"f1_{c.lower()}"] for c in classes) / 3
        
        res["confusion_matrix"] = confusion
        
        res["binary_precision"] = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else 0.0
        res["binary_recall"] = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else 0.0
        res["binary_f1"] = 2 * (res["binary_precision"] * res["binary_recall"]) / (res["binary_precision"] + res["binary_recall"]) if (res["binary_precision"] + res["binary_recall"]) > 0 else 0.0
        res["binary_accuracy"] = (bin_tp + bin_tn) / total if total > 0 else 0.0
        
        return res
