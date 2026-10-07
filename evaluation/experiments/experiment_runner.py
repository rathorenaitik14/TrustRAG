import time
from typing import Dict, Any
from .ablation import AblationStudy

class ExperimentRunner:
    def __init__(self, ablation_study: AblationStudy):
        self.ablation_study = ablation_study
        
    def run_all(self) -> Dict[str, Any]:
        
        # 1. Retrieval
        ret_res = self.ablation_study.run_retrieval_ablation()
        
        # 2. Verification
        ver_res = self.ablation_study.run_verification_ablation()
        preds = ver_res.pop("predictions", [])
        
        # 3. Correction
        cor_res = self.ablation_study.run_correction_ablation(preds)
        
        return {
            "retrieval": ret_res,
            "verification": ver_res,
            "correction": cor_res
        }
        
    def measure_latency(self) -> Dict[str, Dict[str, float]]:
        # Measure latency for FAISS retrieval, CrossEncoder reranking, and NLI verification.
        
        dataset = self.ablation_study.dataset
        
        faiss_times = []
        rerank_times = []
        nli_times = []
        
        # We need a small loop
        for c in dataset:
            claim = c["injected_claim"] or c["original_claims"][0]
            
            # FAISS
            start = time.perf_counter()
            docs = self.ablation_study.retriever.retrieve(claim, k=10)
            faiss_times.append(time.perf_counter() - start)
            
            # Rerank
            start = time.perf_counter()
            reranked = self.ablation_study.reranker.rerank(claim, docs, top_k=5)
            rerank_times.append(time.perf_counter() - start)
            
            # NLI
            # Build mock ev_res
            ev_res = {
                "claim_id": c["case_id"],
                "claim_text": claim,
                "evidence": [
                    {"evidence_id": f"E{i}", "text": r["text"], "document_name": "mock"} 
                    for i, r in enumerate(reranked)
                ]
            }
            start = time.perf_counter()
            self.ablation_study.verifier.verify(ev_res)
            nli_times.append(time.perf_counter() - start)
            
        def agg(times):
            times.sort()
            return {
                "mean_latency": sum(times) / len(times) if times else 0.0,
                "median_latency": times[len(times)//2] if times else 0.0,
                "samples": len(times)
            }
            
        return {
            "FAISS_Retrieval": agg(faiss_times),
            "CrossEncoder_Reranking": agg(rerank_times),
            "NLI_Verification": agg(nli_times)
        }
