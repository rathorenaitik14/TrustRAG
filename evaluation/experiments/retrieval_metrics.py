from typing import List, Dict, Any
from .models import RetrievalMetricsResult

class RetrievalMetricsCalculator:
    def calculate(
        self, 
        ground_truth_map: Dict[str, str], 
        retrieved_results: Dict[str, List[Dict[str, Any]]]
    ) -> RetrievalMetricsResult:
        """
        ground_truth_map: mapping from case_id to expected evidence text/chunk identifier.
        retrieved_results: mapping from case_id to list of retrieved items (e.g. dict with 'text').
        """
        valid_cases = 0
        hits_1, hits_3, hits_5 = 0, 0, 0
        sum_rr = 0.0
        
        for case_id, expected_text in ground_truth_map.items():
            results = retrieved_results.get(case_id, [])
            if expected_text:
                valid_cases += 1
                
                first_relevant_rank = None
                for i, res in enumerate(results):
                    # using basic string inclusion for matching evidence reference
                    if expected_text.lower() in str(res.get("text", "")).lower():
                        first_relevant_rank = i + 1
                        break
                        
                if first_relevant_rank:
                    if first_relevant_rank <= 1: hits_1 += 1
                    if first_relevant_rank <= 3: hits_3 += 1
                    if first_relevant_rank <= 5: hits_5 += 1
                    sum_rr += (1.0 / first_relevant_rank)
                    
        if valid_cases == 0:
            return {"recall_1": 0.0, "recall_3": 0.0, "recall_5": 0.0, "mrr": 0.0}
            
        return {
            "recall_1": hits_1 / valid_cases,
            "recall_3": hits_3 / valid_cases,
            "recall_5": hits_5 / valid_cases,
            "mrr": sum_rr / valid_cases
        }
