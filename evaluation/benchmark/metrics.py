from typing import List, Dict, Any
from .models import BenchmarkCase, BenchmarkMetricsResult

class BenchmarkMetricsCalculator:
    """
    Calculates macro and per-class verification metrics and correction metrics.
    """
    def __init__(self, cases: List[BenchmarkCase], predictions: List[Dict[str, str]]):
        """
        predictions should be a list of dicts with:
        {'case_id': str, 'verification_status': str, 'correction_action': str}
        """
        self.cases = {c["case_id"]: c for c in cases}
        self.predictions = {p["case_id"]: p for p in predictions}
        
    def calculate(self) -> BenchmarkMetricsResult:
        total = len(self.cases)
        
        # We have 3 classes: SUPPORTED, UNSUPPORTED, CONTRADICTED
        classes = ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
        stats = {c: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for c in classes}
        
        correctly_verified = 0
        
        total_problematic = 0
        successfully_corrected = 0
        successfully_removed = 0
        
        total_unsupported = 0
        unsupported_removed = 0
        
        for case_id, case in self.cases.items():
            true_label = case["ground_truth_label"]
            pred = self.predictions.get(case_id, {})
            pred_label = pred.get("verification_status", "UNSUPPORTED")
            
            if true_label == pred_label:
                correctly_verified += 1
                
            for c in classes:
                if true_label == c and pred_label == c:
                    stats[c]["TP"] += 1
                elif true_label != c and pred_label == c:
                    stats[c]["FP"] += 1
                elif true_label == c and pred_label != c:
                    stats[c]["FN"] += 1
                else:
                    stats[c]["TN"] += 1
                    
            # Correction metrics
            if true_label in ["UNSUPPORTED", "CONTRADICTED"]:
                total_problematic += 1
                action = pred.get("correction_action", "")
                if action == "CORRECTED":
                    successfully_corrected += 1
                elif action == "REMOVED_OR_QUALIFIED":
                    successfully_removed += 1
                    
            if true_label == "UNSUPPORTED":
                total_unsupported += 1
                action = pred.get("correction_action", "")
                if action == "REMOVED_OR_QUALIFIED":
                    unsupported_removed += 1
        
        # Calculate P, R, F1 per class
        class_metrics = {}
        for c in classes:
            tp = stats[c]["TP"]
            fp = stats[c]["FP"]
            fn = stats[c]["FN"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[c] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
        macro_p = sum(m["precision"] for m in class_metrics.values()) / 3
        macro_r = sum(m["recall"] for m in class_metrics.values()) / 3
        macro_f1 = sum(m["f1"] for m in class_metrics.values()) / 3
        
        corr_success = successfully_corrected + successfully_removed
        corr_rate = (corr_success / total_problematic) if total_problematic > 0 else 0.0
        unsup_rate = (unsupported_removed / total_unsupported) if total_unsupported > 0 else 0.0
        
        return {
            "total_cases": total,
            "correctly_verified_cases": correctly_verified,
            "verification_accuracy": (correctly_verified / total) if total > 0 else 0.0,
            
            "precision_supported": class_metrics["SUPPORTED"]["precision"],
            "recall_supported": class_metrics["SUPPORTED"]["recall"],
            "f1_supported": class_metrics["SUPPORTED"]["f1"],
            
            "precision_unsupported": class_metrics["UNSUPPORTED"]["precision"],
            "recall_unsupported": class_metrics["UNSUPPORTED"]["recall"],
            "f1_unsupported": class_metrics["UNSUPPORTED"]["f1"],
            
            "precision_contradicted": class_metrics["CONTRADICTED"]["precision"],
            "recall_contradicted": class_metrics["CONTRADICTED"]["recall"],
            "f1_contradicted": class_metrics["CONTRADICTED"]["f1"],
            
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            
            "total_problematic_claims": total_problematic,
            "successfully_corrected": successfully_corrected,
            "successfully_removed_or_qualified": successfully_removed,
            "correction_success_rate": corr_rate,
            
            "unsupported_claims": total_unsupported,
            "removed_or_qualified_unsupported": unsupported_removed,
            "unsupported_claim_removal_rate": unsup_rate
        }
