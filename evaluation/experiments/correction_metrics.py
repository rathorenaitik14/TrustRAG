from typing import List, Dict, Any
from .models import CorrectionMetricsResult

class CorrectionMetricsCalculator:
    def calculate(self, predictions: List[Dict[str, str]]) -> CorrectionMetricsResult:
        """
        predictions: [{'true_label': '...', 'pred_action': '...', 'pred_label': '...'}, ...]
        """
        prob_claims = 0
        successful = 0
        
        uns_claims = 0
        uns_removed = 0
        
        con_claims = 0
        con_corrected = 0
        
        residual = 0
        
        for p in predictions:
            t = p.get("true_label", "")
            action = p.get("pred_action", "")
            # Assume verification pred_label is what triggers the correction logic in reality,
            # but we are evaluating end-to-end success based on true_label vs action.
            if t in ["UNSUPPORTED", "CONTRADICTED"]:
                prob_claims += 1
                
                # Check if it was successfully handled (either corrected or removed)
                if action in ["CORRECTED", "REMOVED_OR_QUALIFIED"]:
                    successful += 1
                else:
                    residual += 1
                    
                if t == "UNSUPPORTED":
                    uns_claims += 1
                    if action == "REMOVED_OR_QUALIFIED":
                        uns_removed += 1
                        
                elif t == "CONTRADICTED":
                    con_claims += 1
                    if action == "CORRECTED":
                        con_corrected += 1
                        
        return {
            "total_problematic_claims": prob_claims,
            "successful_corrections": successful,
            "correction_success_rate": (successful / prob_claims) if prob_claims > 0 else 0.0,
            "total_unsupported_claims": uns_claims,
            "unsupported_claim_removal_rate": (uns_removed / uns_claims) if uns_claims > 0 else 0.0,
            "total_contradicted_claims": con_claims,
            "contradicted_correction_rate": (con_corrected / con_claims) if con_claims > 0 else 0.0,
            "residual_problematic_claims": residual
        }
