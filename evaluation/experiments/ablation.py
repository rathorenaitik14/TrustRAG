import time
from typing import List, Dict, Any, Tuple
from evaluation.benchmark.dataset import get_benchmark_dataset
from evaluation.experiments.retrieval_metrics import RetrievalMetricsCalculator
from evaluation.experiments.verification_metrics import VerificationMetricsCalculator
from evaluation.experiments.correction_metrics import CorrectionMetricsCalculator

class AblationStudy:
    def __init__(
        self,
        retriever=None,
        reranker=None,
        verifier=None,
        evidence_retriever_factory=None # A callable that takes (retriever, reranker) and returns an EvidenceRetriever
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.verifier = verifier
        self.evidence_retriever_factory = evidence_retriever_factory
        self.dataset = get_benchmark_dataset()

    def run_retrieval_ablation(self) -> Dict[str, Any]:
        """A1 vs A2"""
        if not self.retriever or not self.reranker:
            return {}
            
        ground_truth_map = {c["case_id"]: c["evidence_reference"] for c in self.dataset}
        
        # A1: FAISS only
        a1_results = {}
        for c in self.dataset:
            # We mock the Claim representation
            claim = {"claim_id": c["case_id"], "claim_text": c["injected_claim"] or c["original_claims"][0]}
            res = self.retriever.retrieve(claim["claim_text"], k=5)
            # res is list of dict
            a1_results[c["case_id"]] = [{"text": r["text"]} for r in res]
            
        # A2: FAISS + Reranker
        a2_results = {}
        for c in self.dataset:
            claim = {"claim_id": c["case_id"], "claim_text": c["injected_claim"] or c["original_claims"][0]}
            res = self.retriever.retrieve(claim["claim_text"], k=10)
            reranked = self.reranker.rerank(claim["claim_text"], res, top_k=5)
            a2_results[c["case_id"]] = [{"text": r["text"]} for r in reranked]
            
        calc = RetrievalMetricsCalculator()
        return {
            "A1_faiss_only": calc.calculate(ground_truth_map, a1_results),
            "A2_faiss_rerank": calc.calculate(ground_truth_map, a2_results)
        }

    def run_verification_ablation(self) -> Dict[str, Any]:
        """A2 vs A3 - This primarily shows the Verification stats."""
        if not self.evidence_retriever_factory or not self.verifier:
            return {}
            
        ev_ret = self.evidence_retriever_factory(self.retriever, self.reranker)
        
        predictions = []
        for c in self.dataset:
            claim = {"claim_id": c["case_id"], "claim_text": c["injected_claim"] or c["original_claims"][0]}
            ev_res = ev_ret.retrieve_for_claims([claim])[0]
            vr = self.verifier.verify(ev_res)
            
            predictions.append({
                "case_id": c["case_id"],
                "true_label": c["ground_truth_label"],
                "pred_label": vr["status"]
            })
            
        calc = VerificationMetricsCalculator()
        res = calc.calculate(predictions)
        return {"A3_verification": res, "predictions": predictions}

    def run_correction_ablation(self, predictions: List[Dict[str, str]]) -> Dict[str, Any]:
        """A3 vs A4"""
        
        # We mock the correction layer's behavior based on the verifier's prediction
        # just like we did in the benchmark runner, to avoid LLM cost in deterministic runs.
        corr_preds = []
        for p in predictions:
            pr = p["pred_label"]
            action = "PRESERVED"
            if pr == "CONTRADICTED":
                action = "CORRECTED"
            elif pr == "UNSUPPORTED":
                action = "REMOVED_OR_QUALIFIED"
                
            corr_preds.append({
                "true_label": p["true_label"],
                "pred_label": pr,
                "pred_action": action
            })
            
        calc = CorrectionMetricsCalculator()
        return {"A4_correction": calc.calculate(corr_preds)}
