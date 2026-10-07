from typing import List, Dict, Any, Optional
from .models import BenchmarkCase, BenchmarkMetricsResult
from .metrics import BenchmarkMetricsCalculator

class BenchmarkRunner:
    def __init__(
        self,
        extractor=None,
        evidence_retriever=None,
        verifier=None,
        corrector=None,
        report_generator=None
    ):
        self.extractor = extractor
        self.evidence_retriever = evidence_retriever
        self.verifier = verifier
        self.corrector = corrector
        self.report_generator = report_generator
        
    def run(self, cases: List[BenchmarkCase], use_live_llm: bool = False) -> Dict[str, Any]:
        predictions = []
        reports = []
        
        for case in cases:
            case_id = case["case_id"]
            answer = case["injected_answer"]
            
            # 1. Extraction
            if use_live_llm and self.extractor:
                ext_res = self.extractor.extract_claims(answer, case["question"])
                claims = ext_res["claims"]
            else:
                # Deterministic bypass using dataset claims
                claims = []
                for i, c in enumerate(case["original_claims"]):
                    claims.append({"claim_id": f"C{i+1}", "claim_text": c})
                if case["injected_claim"]:
                    claims.append({"claim_id": f"C{len(claims)+1}", "claim_text": case["injected_claim"]})
                    
            # Identify the injected claim ID to track its status
            injected_claim_id = None
            if case["injected_claim"]:
                for c in claims:
                    if c["claim_text"] == case["injected_claim"]:
                        injected_claim_id = c["claim_id"]
                        break
            else:
                # If no injected claim, track the first claim (SUPPORTED cases)
                injected_claim_id = claims[0]["claim_id"] if claims else None
                
            # 2. Evidence Retrieval
            if self.evidence_retriever:
                evidence_results = self.evidence_retriever.retrieve_for_claims(claims)
            else:
                # Mock empty evidence for testing logic if retriever absent
                evidence_results = [{"claim_id": c["claim_id"], "evidence": []} for c in claims]
                
            # 3. Verification
            verification_results = []
            if self.verifier:
                for ev in evidence_results:
                    verification_results.append(self.verifier.verify(ev))
            else:
                # Mock verification using ground truth for testing
                for c in claims:
                    status = case["ground_truth_label"] if c["claim_id"] == injected_claim_id else "SUPPORTED"
                    verification_results.append({
                        "claim_id": c["claim_id"],
                        "claim_text": c["claim_text"],
                        "status": status,
                        "score": 0.99,
                        "evidence_ids": []
                    })
                    
            # Find status for the injected/target claim
            pred_status = "UNSUPPORTED"
            pred_score = 0.0
            for vr in verification_results:
                if vr["claim_id"] == injected_claim_id:
                    pred_status = vr["status"]
                    pred_score = vr.get("score", 0.0)
                    break
                    
            # 4. Correction
            correction_changes = []
            if use_live_llm and self.corrector:
                corr_res = self.corrector.correct(answer, claims, evidence_results, verification_results)
                correction_changes = corr_res["changes"]
            else:
                for vr in verification_results:
                    action = "PRESERVED"
                    if vr["status"] == "CONTRADICTED":
                        action = "CORRECTED"
                    elif vr["status"] == "UNSUPPORTED":
                        action = "REMOVED_OR_QUALIFIED"
                    correction_changes.append({"claim_id": vr["claim_id"], "action": action})
                    
            pred_action = "PRESERVED"
            for ch in correction_changes:
                if ch["claim_id"] == injected_claim_id:
                    pred_action = ch["action"]
                    break
                    
            predictions.append({
                "case_id": case_id,
                "verification_status": pred_status,
                "correction_action": pred_action,
                "verification_score": pred_score
            })
            
            # 5. Report
            if self.report_generator:
                rep = self.report_generator.generate_report(
                    question=case["question"],
                    original_answer=answer,
                    corrected_answer="Mock corrected." if not use_live_llm else corr_res.get("corrected_answer", ""),
                    extracted_claims=claims,
                    evidence_results=evidence_results,
                    verification_results=verification_results,
                    correction_changes=correction_changes
                )
                reports.append(rep)
                
        calc = BenchmarkMetricsCalculator(cases, predictions)
        metrics = calc.calculate()
        
        return {
            "metrics": metrics,
            "predictions": predictions,
            "reports": reports
        }
