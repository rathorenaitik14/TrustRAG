import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv

from claims.claim_extractor import ClaimExtractor
from claims.evidence_retriever import EvidenceRetriever
from verification.nli_verifier import NLIVerifier
from correction.gemini_corrector import GeminiCorrector
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
    
    print("Initializing LLM Client...")
    try:
        llm_client = get_llm_client()
    except Exception as e:
        print("Skipping LLM init due to config missing.")
        sys.exit(1)
        
    extractor = ClaimExtractor(client=llm_client)
    evidence_retriever = EvidenceRetriever(retriever, reranker, default_evidence_top_k=2)
    nli_verifier = NLIVerifier(model_name="cross-encoder/nli-deberta-v3-small")
    corrector = GeminiCorrector(client=llm_client)

    test_cases = [
        {
            "question": "What is the goal of data mining?",
            "answer": "The goal of data mining is to provide efficient tools and techniques for KDD. Data mining was invented in 1970."
        },
        {
            "question": "CONTRADICTION TEST",
            "answer": "The mean is always the best measure of central tendency."
        },
        {
            "question": "UNSUPPORTED TEST",
            "answer": "Data mining was invented at MIT."
        }
    ]
    
    pre_extracted = {
        "What is the goal of data mining?": [
            {"claim_id": "C1", "claim_text": "The goal of data mining is to provide efficient tools and techniques for KDD.", "claim_type": "factual"},
            {"claim_id": "C2", "claim_text": "Data mining was invented in 1970.", "claim_type": "factual"}
        ],
        "CONTRADICTION TEST": [
            {"claim_id": "C1", "claim_text": "The mean is always the best measure of central tendency.", "claim_type": "factual"}
        ],
        "UNSUPPORTED TEST": [
            {"claim_id": "C1", "claim_text": "Data mining was invented at MIT.", "claim_type": "factual"}
        ]
    }

    for i, tc in enumerate(test_cases, 1):
        print(f"\n=======================================================")
        print(f"TEST CASE {i}")
        print(f"=======================================================\n")
        
        q = tc["question"]
        a = tc["answer"]
        print(f"ORIGINAL ANSWER:\n{a}\n")
        
        # 1. Claims
        claims_list = pre_extracted[q]
            
        # 2. Evidence
        evidence_results = evidence_retriever.retrieve_for_claims(claims_list)
        
        # 3. Verification
        verification_results = []
        print("CLAIM VERIFICATION:\n")
        for cr in evidence_results:
            vr = nli_verifier.verify(cr)
            verification_results.append(vr)
            print(f"{vr['claim_id']} -> {vr['status']}")
            
        print("\nRunning Corrector (LLM)...\n")
        
        # 4. Correction
        try:
            correction_res = corrector.correct(a, claims_list, evidence_results, verification_results)
            print(f"CORRECTED ANSWER:\n{correction_res['corrected_answer']}\n")
            print("CHANGES:\n")
            for change in correction_res["changes"]:
                print(f"{change['claim_id']} -> {change['action']}")
        except Exception as e:
            print(f"Correction failed: {e}")

if __name__ == "__main__":
    main()
