import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from claims.claim_extractor import ClaimExtractor
from claims.evidence_retriever import EvidenceRetriever
from verification.nli_verifier import NLIVerifier
from generation.llm_client import get_llm_client
from retrieval.retriever import RAGRetriever
from retrieval.reranker import CrossEncoderReranker

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    load_dotenv()
    
    base_dir = Path("data/processed")
    jsonl_files = list(base_dir.glob("*_chunks.jsonl"))
    if not jsonl_files:
        print("No chunk files found.")
        sys.exit(1)
        
    target_jsonl = next((f for f in jsonl_files if "DEDV" in f.name), jsonl_files[0])
    faiss_path = target_jsonl.with_name(f"{target_jsonl.stem}_faiss.index")
    metadata_path = target_jsonl.with_name(f"{target_jsonl.stem}_embeddings_metadata.json")
    chunks_path = str(target_jsonl)
    
    print("Initializing Retriever...")
    retriever = RAGRetriever(str(faiss_path), str(metadata_path), chunks_path)
    
    print("Initializing Reranker...")
    reranker = CrossEncoderReranker()
    
    print("Initializing Claim Extractor...")
    try:
        llm_client = get_llm_client()
        extractor = ClaimExtractor(client=llm_client)
    except Exception as e:
        extractor = None
        
    print("Initializing Evidence Retriever...")
    evidence_retriever = EvidenceRetriever(retriever, reranker, default_evidence_top_k=2)

    print("Initializing NLI Verifier (cross-encoder/nli-deberta-v3-small)...")
    nli_verifier = NLIVerifier(model_name="cross-encoder/nli-deberta-v3-small")

    # Static answers to avoid 429 errors from the Generator step
    test_cases = [
        {
            "question": "What is the goal of data mining?",
            "answer": "The goal of data mining is to provide efficient tools and techniques for Knowledge Discovery in Databases (KDD)."
        },
        {
            "question": "What is KDD?",
            "answer": "KDD stands for Knowledge Discovery in Databases. It is the overall process of discovering useful knowledge from data."
        },
        {
            "question": "CONTROLLED FALSE-CLAIM DEMONSTRATION",
            "answer": "Data mining was invented in 1970."
        }
    ]
    
    pre_extracted = {
        "What is the goal of data mining?": [
            {"claim_id": "C1", "claim_text": "The goal of data mining is to provide efficient tools and techniques for Knowledge Discovery in Databases (KDD).", "claim_type": "factual"}
        ],
        "What is KDD?": [
            {"claim_id": "C1", "claim_text": "KDD stands for Knowledge Discovery in Databases.", "claim_type": "factual"}
        ],
        "CONTROLLED FALSE-CLAIM DEMONSTRATION": [
            {"claim_id": "C1", "claim_text": "Data mining was invented in 1970.", "claim_type": "factual"}
        ]
    }

    for i, tc in enumerate(test_cases, 1):
        print(f"\n=======================================================")
        print(f"TEST CASE {i}")
        print(f"=======================================================\n")
        
        q = tc["question"]
        a = tc["answer"]
        print(f"QUESTION:\n{q}\n")
        
        # 1. Extract claims (with fallback)
        claims_list = []
        if extractor:
            for attempt in range(2):
                try:
                    res = extractor.extract_claims(a, question=q)
                    claims_list = res["claims"]
                    break
                except Exception:
                    time.sleep(1)
            if not claims_list:
                claims_list = pre_extracted[q]
        else:
            claims_list = pre_extracted[q]
            
        # 2. Retrieve evidence
        evidence_results = evidence_retriever.retrieve_for_claims(claims_list)
        
        # 3. Verify
        for cr in evidence_results:
            print(f"CLAIM {cr['claim_id']}:\n{cr['claim_text']}\n")
            
            verification_result = nli_verifier.verify(cr)
            
            print("EVIDENCE & NLI:")
            for ev_det in verification_result["evidence_details"]:
                ev_id = ev_det["evidence_id"]
                ev_text = next((e["text"] for e in cr["evidence"] if e["evidence_id"] == ev_id), "N/A")
                print(f"  {ev_id}")
                print(f"  Text: {ev_text}")
                print(f"  NLI Status: {ev_det['status']} (Score: {ev_det['score']:.4f})\n")
                
            print(f"FINAL STATUS: {verification_result['status']}")
            print(f"AGGREGATED SCORE: {verification_result['score']:.4f}")
            print(f"REASON: {verification_result['reason']}\n")

if __name__ == "__main__":
    main()
