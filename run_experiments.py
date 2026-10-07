import sys
from pathlib import Path
from evaluation.experiments.ablation import AblationStudy
from evaluation.experiments.experiment_runner import ExperimentRunner
from claims.evidence_retriever import EvidenceRetriever
from verification.nli_verifier import NLIVerifier
from retrieval.retriever import RAGRetriever
from retrieval.reranker import CrossEncoderReranker

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    base_dir = Path("data/processed")
    jsonl_files = list(base_dir.glob("*_chunks.jsonl"))
    if not jsonl_files:
        print("No chunk files found.")
        sys.exit(1)
        
    target_jsonl = next((f for f in jsonl_files if "DEDV" in f.name), jsonl_files[0])
    faiss_path = target_jsonl.with_name(f"{target_jsonl.stem}_faiss.index")
    metadata_path = target_jsonl.with_name(f"{target_jsonl.stem}_embeddings_metadata.json")
    chunks_path = str(target_jsonl)
    
    print("Loading Models for Experiment...")
    retriever = RAGRetriever(str(faiss_path), str(metadata_path), chunks_path)
    reranker = CrossEncoderReranker()
    verifier = NLIVerifier(model_name="cross-encoder/nli-deberta-v3-small")
    
    def er_factory(ret, rer):
        return EvidenceRetriever(ret, rer, default_evidence_top_k=2)
        
    study = AblationStudy(
        retriever=retriever,
        reranker=reranker,
        verifier=verifier,
        evidence_retriever_factory=er_factory
    )
    
    runner = ExperimentRunner(study)
    
    print("Running Experiments...")
    results = runner.run_all()
    
    print("Measuring Latency...")
    latency = runner.measure_latency()
    
    ret = results["retrieval"]
    ver = results["verification"]["A3_verification"]
    cor = results["correction"]["A4_correction"]
    
    print("\n==================================================")
    print("TRUSTRAG EXPERIMENTAL EVALUATION")
    print("==================================================")
    print("\nDATASET")
    print(f"Name: {study.dataset[0]['document_name']}")
    print(f"Sample Size: {len(study.dataset)} cases")
    print("\nNote: The current benchmark contains 15 manually constructed cases and therefore provides a preliminary evaluation rather than statistically conclusive evidence.\n")
    
    print("--------------------------------------------------")
    print("EXPERIMENT 1 - RETRIEVAL")
    print("--------------------------------------------------")
    
    a1 = ret["A1_faiss_only"]
    print("\nFAISS ONLY")
    print(f"Recall@1: {a1['recall_1']:.4f}")
    print(f"Recall@3: {a1['recall_3']:.4f}")
    print(f"Recall@5: {a1['recall_5']:.4f}")
    print(f"MRR:      {a1['mrr']:.4f}")
    
    a2 = ret["A2_faiss_rerank"]
    print("\nFAISS + CROSSENCODER")
    print(f"Recall@1: {a2['recall_1']:.4f}")
    print(f"Recall@3: {a2['recall_3']:.4f}")
    print(f"Recall@5: {a2['recall_5']:.4f}")
    print(f"MRR:      {a2['mrr']:.4f}")
    
    print("\n--------------------------------------------------")
    print("EXPERIMENT 2 - VERIFICATION")
    print("--------------------------------------------------")
    
    print(f"\nAccuracy:       {ver['accuracy']:.4f}")
    print(f"Macro Precision: {ver['macro_precision']:.4f}")
    print(f"Macro Recall:    {ver['macro_recall']:.4f}")
    print(f"Macro F1:        {ver['macro_f1']:.4f}")
    
    print("\nPer-class results:")
    print("\nSUPPORTED:")
    print(f"Precision: {ver['precision_supported']:.4f}")
    print(f"Recall:    {ver['recall_supported']:.4f}")
    print(f"F1:        {ver['f1_supported']:.4f}")
    
    print("\nUNSUPPORTED:")
    print(f"Precision: {ver['precision_unsupported']:.4f}")
    print(f"Recall:    {ver['recall_unsupported']:.4f}")
    print(f"F1:        {ver['f1_unsupported']:.4f}")
    
    print("\nCONTRADICTED:")
    print(f"Precision: {ver['precision_contradicted']:.4f}")
    print(f"Recall:    {ver['recall_contradicted']:.4f}")
    print(f"F1:        {ver['f1_contradicted']:.4f}")
    
    print("\n--------------------------------------------------")
    print("EXPERIMENT 3 - CORRECTION")
    print("--------------------------------------------------")
    
    print(f"\nProblematic claims:             {cor['total_problematic_claims']}")
    print(f"Successful corrections:         {cor['successful_corrections']}")
    print(f"Correction Success Rate:        {cor['correction_success_rate']:.4f}")
    print(f"Unsupported Claim Removal Rate: {cor['unsupported_claim_removal_rate']:.4f}")
    print(f"Contradicted Correction Rate:   {cor['contradicted_correction_rate']:.4f}")
    print(f"Residual Problematic Claims:    {cor['residual_problematic_claims']}")
    
    print("\n--------------------------------------------------")
    print("EXPERIMENT 4 - ABLATION")
    print("--------------------------------------------------")
    
    print("\nConfiguration")
    print("Metrics")
    
    print("\nA1 FAISS only")
    print(f"Retrieval MRR: {a1['mrr']:.4f}")
    
    print("\nA2 FAISS + CrossEncoder")
    print(f"Retrieval MRR: {a2['mrr']:.4f}")
    
    print("\nA3 + Verification")
    print(f"Verification Accuracy: {ver['accuracy']:.4f}")
    print(f"Problematic Claims Detected: {ver['confusion_matrix']['UNSUPPORTED']['UNSUPPORTED'] + ver['confusion_matrix']['CONTRADICTED']['CONTRADICTED']}")
    
    print("\nA4 + Correction")
    print(f"Residual Problematic Claims: {cor['residual_problematic_claims']}")
    print(f"Correction Success Rate: {cor['correction_success_rate']:.4f}")
    
    print("\n--------------------------------------------------")
    print("LATENCY MEASUREMENTS (Local deterministic models only)")
    print("--------------------------------------------------")
    
    for k, v in latency.items():
        print(f"\n{k}:")
        print(f"Mean Latency:   {v['mean_latency']:.4f}s")
        print(f"Median Latency: {v['median_latency']:.4f}s")
        print(f"Samples:        {v['samples']}")
        
    print("\n==================================================")
    
if __name__ == "__main__":
    main()
