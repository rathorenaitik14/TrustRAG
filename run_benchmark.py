import sys
from pathlib import Path

from evaluation.benchmark.dataset import get_benchmark_dataset
from evaluation.benchmark.runner import BenchmarkRunner
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
    
    print("Loading Local Models (0 API cost)...")
    retriever = RAGRetriever(str(faiss_path), str(metadata_path), chunks_path)
    reranker = CrossEncoderReranker()
    evidence_retriever = EvidenceRetriever(retriever, reranker, default_evidence_top_k=2)
    nli_verifier = NLIVerifier(model_name="cross-encoder/nli-deberta-v3-small")
    
    runner = BenchmarkRunner(
        extractor=None,  # Bypass LLM extraction
        evidence_retriever=evidence_retriever,
        verifier=nli_verifier,
        corrector=None,  # Bypass LLM correction
        report_generator=None
    )
    
    dataset = get_benchmark_dataset()
    print(f"\nRunning benchmark on {len(dataset)} cases...\n")
    
    results = runner.run(dataset, use_live_llm=False)
    metrics = results["metrics"]
    predictions = results["predictions"]
    
    sup = len([c for c in dataset if c["ground_truth_label"] == "SUPPORTED"])
    uns = len([c for c in dataset if c["ground_truth_label"] == "UNSUPPORTED"])
    con = len([c for c in dataset if c["ground_truth_label"] == "CONTRADICTED"])
    
    print("==================================================")
    print("TRUSTRAG CONTROLLED BENCHMARK")
    print("==================================================")
    print(f"\nDataset: DEDV Unit1_Data Preprocessing An Overview.pdf")
    print(f"Total cases: {metrics['total_cases']}")
    print(f"\nSUPPORTED:\n{sup}")
    print(f"UNSUPPORTED:\n{uns}")
    print(f"CONTRADICTED:\n{con}")
    
    print("\n--------------------------------------------------")
    print("VERIFICATION RESULTS")
    print("--------------------------------------------------")
    print(f"\nAccuracy:\n{metrics['verification_accuracy']:.4f}")
    print(f"\nMacro Precision:\n{metrics['macro_precision']:.4f}")
    print(f"Macro Recall:\n{metrics['macro_recall']:.4f}")
    print(f"Macro F1:\n{metrics['macro_f1']:.4f}")
    
    print("\n--------------------------------------------------")
    print("CORRECTION RESULTS (MOCKED DETERMINISTIC)")
    print("--------------------------------------------------")
    print(f"\nProblematic Claims:\n{metrics['total_problematic_claims']}")
    print(f"Corrected:\n{metrics['successfully_corrected']}")
    print(f"Removed/Qualified:\n{metrics['successfully_removed_or_qualified']}")
    print(f"\nCorrection Success Rate:\n{metrics['correction_success_rate']:.4f}")
    print(f"Unsupported Claim Removal Rate:\n{metrics['unsupported_claim_removal_rate']:.4f}")
    
    print("\n==================================================")
    print("BENCHMARK CASES")
    print("==================================================")
    
    for case, pred in zip(dataset, predictions):
        print(f"\nCase ID: {case['case_id']}")
        print(f"Question: {case['question']}")
        print(f"Injected Claim: {case['injected_claim'] if case['injected_claim'] else 'None'}")
        print(f"Ground Truth: {case['ground_truth_label']}")
        print(f"Predicted Status: {pred['verification_status']}")
        print(f"Verification Score: {pred['verification_score']:.4f}")
        print(f"Correction Action: {pred['correction_action']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
