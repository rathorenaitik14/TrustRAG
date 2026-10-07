from typing import List, Dict, Any
import copy

class ContextBuilder:
    def __init__(self, top_k: int = 5):
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        self.top_k = top_k

    def build_context(self, question: str, reranked_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Builds a formatted context from the user's question and reranked retrieval results.
        Returns a dictionary containing the question, the structured evidence list, and the formatted text.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")
            
        # Do not modify the original results
        results_copy = copy.deepcopy(reranked_results)
        
        # Limit to top_k
        results_to_use = results_copy[:self.top_k]
        
        evidence_list = []
        formatted_context_parts = []
        
        formatted_context_parts.append(f"QUESTION:\n{question}\n")
        
        for idx, result in enumerate(results_to_use, start=1):
            # Validate required fields
            required_fields = [
                "chunk_id", "document_id", "document_name", "page_start", 
                "page_end", "text", "retrieval_rank", "retrieval_score", 
                "rerank_rank", "rerank_score", "source_path"
            ]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Retrieval record is missing required field: '{field}'")
            
            evidence_id = f"E{idx}"
            
            evidence_item = {
                "evidence_id": evidence_id,
                "chunk_id": result["chunk_id"],
                "document_id": result["document_id"],
                "document_name": result["document_name"],
                "page_start": result["page_start"],
                "page_end": result["page_end"],
                "section": result.get("section"),
                "source_path": result["source_path"],
                "retrieval_rank": result["retrieval_rank"],
                "retrieval_score": result["retrieval_score"],
                "rerank_rank": result["rerank_rank"],
                "rerank_score": result["rerank_score"],
                "text": result["text"]
            }
            evidence_list.append(evidence_item)
            
            # Format the text block
            pages_str = f"{evidence_item['page_start']}–{evidence_item['page_end']}" if evidence_item['page_start'] != evidence_item['page_end'] else str(evidence_item['page_start'])
            
            block = (
                f"EVIDENCE {evidence_id}\n"
                f"Source: {evidence_item['document_name']}\n"
                f"Pages: {pages_str}\n"
                f"Chunk ID: {evidence_item['chunk_id']}\n"
                f"FAISS Rank: {evidence_item['retrieval_rank']}\n"
                f"FAISS Score: {evidence_item['retrieval_score']:.4f}\n"
                f"Rerank Rank: {evidence_item['rerank_rank']}\n"
                f"CrossEncoder Score: {evidence_item['rerank_score']:.4f}\n\n"
                f"{evidence_item['text']}\n"
            )
            formatted_context_parts.append(block)
            
            if idx < len(results_to_use):
                formatted_context_parts.append("-" * 50 + "\n")
                
        # Append grounding instructions
        instructions = (
            "\nUse only the provided evidence to answer the question.\n"
            "Do not introduce factual information that is not supported\n"
            "by the provided evidence.\n"
            "If the evidence is insufficient to answer the question,\n"
            "state that the available evidence is insufficient."
        )
        
        if not results_to_use:
            formatted_context_parts.append("No evidence provided.")
            
        formatted_context_parts.append(instructions)
        
        formatted_context = "\n".join(formatted_context_parts)
        
        return {
            "question": question,
            "evidence_list": evidence_list,
            "formatted_context": formatted_context
        }
