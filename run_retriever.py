import os
from pathlib import Path
from retrieval.retriever import RAGRetriever

def main():
    base_dir = Path("data/processed")
    
    # Identify the files
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
        
    print("Initializing RAGRetriever...")
    retriever = RAGRetriever(
        faiss_index_path=str(faiss_path),
        metadata_path=str(meta_path),
        chunks_jsonl_path=str(target_jsonl),
        default_top_k=5
    )
    print("RAGRetriever successfully initialized.")
    
    test_queries = [
        "What is the goal of data mining?",
        "What is data preprocessing?",
        "What is KDD?",
        "What is the data engineering lifecycle?",
        "What are measures of central tendency?"
    ]
    
    for i, question in enumerate(test_queries, 1):
        print(f"\n=======================================================")
        print(f"QUESTION: {question}")
        print(f"=======================================================\n")
        print("TOP 5 RETRIEVED EVIDENCE\n")
        
        results = retriever.retrieve(question, k=5)
        
        for result in results:
            rank = result["rank"]
            score = result["score"]
            chunk_id = result["chunk_id"]
            doc_name = result["document_name"]
            page = f"{result['page_start']}-{result['page_end']}"
            section = result.get("section") or "None"
            text = result["text"]
            
            print(f"Rank: {rank}")
            print(f"Score: {score:.4f}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Document: {doc_name}")
            print(f"Page: {page}")
            print(f"Section: {section}")
            
            preview = text.replace('\n', ' ')[:150] + "..." if len(text) > 150 else text.replace('\n', ' ')
            print(f"Text preview: {preview}\n")

if __name__ == "__main__":
    main()
