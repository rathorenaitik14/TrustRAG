import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from claims.claim_extractor import ClaimExtractor
from claims.evidence_retriever import EvidenceRetriever
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
        print(f"Skipping LLM init due to config missing: {e}")
        extractor = None
        
    print("Initializing Evidence Retriever...")
    evidence_retriever = EvidenceRetriever(retriever, reranker, default_evidence_top_k=2)

    # Static answers to avoid 429 errors from the Generator step
    test_cases = [
        {
            "question": "What is the goal of data mining?",
            "answer": "The goal of data mining is to provide efficient tools and techniques for Knowledge Discovery in Databases (KDD)."
        },
        {
            "question": "CONTROLLED FALSE-CLAIM TEST",
            "answer": "Data mining was invented in 1970."
        }
    ]
    
    # We will use pre-extracted claims just in case the Extractor hits 429 quota exhaustion too.
    pre_extracted = {
        "What is the goal of data mining?": [
            {"claim_id": "C1", "claim_text": "The goal of data mining is to provide efficient tools and techniques for Knowledge Discovery in Databases (KDD).", "claim_type": "factual"}
        ],
        "CONTROLLED FALSE-CLAIM TEST": [
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
            for attempt in range(4):
                try:
                    res = extractor.extract_claims(a, question=q)
                    claims_list = res["claims"]
                    break
                except Exception as e:
                    if ("429" in str(e) or "503" in str(e)) and attempt < 3:
                        time.sleep(2)
                        continue
                    print("API quota hit during extraction. Using fallback static claims.")
                    claims_list = pre_extracted[q]
                    break
        else:
            claims_list = pre_extracted[q]
            
        if not claims_list:
            continue
            
        # 2. Retrieve evidence for claims
        evidence_results = evidence_retriever.retrieve_for_claims(claims_list)
        
        for cr in evidence_results:
            print(f"CLAIM {cr['claim_id']}:\n{cr['claim_text']}\n")
            print("EVIDENCE:\n")
            for ev in cr["evidence"]:
                print(ev["evidence_id"])
                print(f"Document: {ev['document_name']}")
                print(f"Pages: {ev['page_start']}-{ev.get('page_end', ev['page_start'])}")
                print(f"FAISS Score: {ev['faiss_score']:.4f}")
                print(f"CrossEncoder Score: {ev['rerank_score']:.4f}")
                print(f"Text: {ev['text']}\n")

if __name__ == "__main__":
    main()
