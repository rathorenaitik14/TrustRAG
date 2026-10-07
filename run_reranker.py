import os
from pathlib import Path
from retrieval.retriever import RAGRetriever
from retrieval.reranker import CrossEncoderReranker

def main():
    base_dir = Path("data/processed")
    
    # 1. Setup paths
    jsonl_files = list(base_dir.glob("*_chunks.jsonl"))
    if not jsonl_files:
        print("No chunk files found.")
        return
        
    target_jsonl = next((f for f in jsonl_files if "DEDV" in f.name), jsonl_files[0])
    faiss_path = target_jsonl.with_name(f"{target_jsonl.stem}_faiss.index")
    meta_path = target_jsonl.with_name(f"{target_jsonl.stem}_embeddings_metadata.json")
    
    if not faiss_path.exists() or not meta_path.exists():
        print("FAISS index or metadata not found. Run earlier steps.")
        return
        
    # 2. Initialize Retriever
    print("Initializing Retriever...")
    retriever = RAGRetriever(
        faiss_index_path=str(faiss_path),
        metadata_path=str(meta_path),
        chunks_jsonl_path=str(target_jsonl),
        default_top_k=10
    )
    
    # 3. Initialize Reranker
    print("Initializing Reranker...")
    reranker = CrossEncoderReranker()
    
    test_queries = [
        "What is the goal of data mining?",
        "What is data preprocessing?",
        "What is KDD?",
        "What is the data engineering lifecycle?",
        "What are measures of central tendency?"
    ]
    
    for i, question in enumerate(test_queries, 1):
        print(f"\n=======================================================")
        print(f"QUESTION {i}: {question}")
        print(f"=======================================================")
        
        # STEP A: Retrieve top 10
        candidates = retriever.retrieve(question, k=10)
        
        print("\n--- BEFORE RERANKING (Top 10) ---")
        for cand in candidates:
            rank = cand["rank"]
            score = cand["score"]
            chunk_id = cand["chunk_id"]
            page = f"{cand['page_start']}-{cand['page_end']}"
            text = cand["text"]
            preview = text.replace('\n', ' ')[:100] + "..." if len(text) > 100 else text.replace('\n', ' ')
            print(f"Rank {rank} | Score: {score:.4f} | Chunk ID: {chunk_id} | Page: {page} | Preview: {preview}")
            
        # STEP B: Rerank top 10 to top 5
        reranked = reranker.rerank(question, candidates, top_k=5)
        
        print("\n--- AFTER RERANKING (Top 5) ---")
        for cand in reranked:
            r_rank = cand["rerank_rank"]
            r_score = cand["rerank_score"]
            orig_rank = cand["retrieval_rank"]
            orig_score = cand["retrieval_score"]
            chunk_id = cand["chunk_id"]
            page = f"{cand['page_start']}-{cand['page_end']}"
            text = cand["text"]
            preview = text.replace('\n', ' ')[:100] + "..." if len(text) > 100 else text.replace('\n', ' ')
            
            print(f"Rerank Rank {r_rank} | CE Score: {r_score:.4f} | Orig Rank: {orig_rank} | Orig FAISS: {orig_score:.4f} | Chunk ID: {chunk_id} | Page: {page} | Preview: {preview}")
            
if __name__ == "__main__":
    main()
