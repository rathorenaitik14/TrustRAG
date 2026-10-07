from typing import Dict, Any, Optional
import copy
from .llm_client import LLMClient, get_llm_client

class LLMGenerator:
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_llm_client()

    def generate_answer(self, context_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates an answer using the provided ContextBuilder output.
        """
        if "question" not in context_result or not context_result["question"]:
            raise ValueError("Context is missing a valid 'question'.")
            
        if "formatted_context" not in context_result or "evidence_list" not in context_result:
            raise ValueError("Context result is missing required formatted_context or evidence_list fields.")
            
        # 1. Preserve original context without mutation
        safe_context = copy.deepcopy(context_result)
        
        # 2. System prompt
        system_prompt = (
            "You are a helpful, factual assistant.\n"
            "Answer the user's question using ONLY the supplied evidence.\n"
            "Do not introduce factual information that is not supported by the evidence.\n"
            "If the evidence is insufficient to answer the question, explicitly state that the available evidence is insufficient.\n"
            "Do not invent citations or sources.\n"
            "Do not claim information comes from the documents unless it is actually present in the supplied evidence.\n"
            "Give a concise, direct answer.\n"
            "Preserve important numbers, definitions, names, and terminology from the evidence.\n"
            "Evidence blocks contain provenance such as EVIDENCE E1. When referencing evidence, use only evidence identifiers that actually exist in the supplied context."
        )
        
        # 3. User prompt
        user_prompt = (
            f"QUESTION:\n{safe_context['question']}\n\n"
            f"CONTEXT:\n{safe_context['formatted_context']}"
        )
        
        # 4. Generate answer
        answer = self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        
        # 5. Extract evidence IDs for return format
        evidence_ids = [e["evidence_id"] for e in safe_context["evidence_list"] if "evidence_id" in e]
        
        # 6. Return structured result
        return {
            "question": safe_context["question"],
            "answer": answer,
            "model_name": self.client.model_name,
            "provider": self.client.provider,
            "evidence_ids": evidence_ids
        }
