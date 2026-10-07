import os
import sys
from pathlib import Path
from retrieval.retriever import RAGRetriever
from retrieval.reranker import CrossEncoderReranker
from context.context_builder import ContextBuilder

# Force UTF-8 for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    base_dir = Path("data/processed")
    
    # Setup paths
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
        
    print("Initializing Retriever...")
    retriever = RAGRetriever(
        faiss_index_path=str(faiss_path),
        metadata_path=str(meta_path),
        chunks_jsonl_path=str(target_jsonl),
        default_top_k=10
    )
    
    print("Initializing Reranker...")
    reranker = CrossEncoderReranker()
    
    print("Initializing Context Builder...")
    context_builder = ContextBuilder(top_k=5)
    
    test_queries = [
        "What is the goal of data mining?",
        "What is data preprocessing?",
        "What is KDD?",
        "What is the data engineering lifecycle?",
        "What are measures of central tendency?"
    ]
    
    for i, question in enumerate(test_queries, 1):
        print(f"\n=======================================================")
        print(f"TEST QUESTION {i}")
        print(f"=======================================================\n")
        
        # Retrieval (Top 10)
        candidates = retriever.retrieve(question, k=10)
        
        # Reranking
        reranked = reranker.rerank(question, candidates, top_k=5)
        
        # Context Engineering
        context_result = context_builder.build_context(question, reranked)
        
        print(context_result["formatted_context"])

if __name__ == "__main__":
    main()
