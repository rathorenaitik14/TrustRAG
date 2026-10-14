import json
import re
from typing import Optional, Dict, Any, List
from .models import ClaimExtractionResult

from generation.llm_client import LLMClient, get_llm_client

class ClaimExtractor:
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_llm_client()
        
    def extract_claims(self, answer: str, question: Optional[str] = None) -> ClaimExtractionResult:
        if not answer or not str(answer).strip():
            raise ValueError("The generated answer cannot be empty.")
            
        system_prompt = (
            "You are a strict JSON extraction system.\n"
            "Your task is to extract factual claims from the supplied answer.\n"
            "Follow these strict rules:\n"
            "1. Extract only information explicitly asserted by the answer.\n"
            "2. Do not introduce facts that are not present in the answer.\n"
            "3. Split compound factual statements into atomic claims where practical, but do not destroy the semantic meaning.\n"
            "4. Preserve important names, numbers, dates, definitions, and terminology exactly.\n"
            "5. Do not convert opinions, recommendations, or questions into factual claims.\n"
            "6. Do not extract greetings or conversational filler.\n"
            "7. Do NOT perform verification. Do not decide whether a claim is true or false. Extract it exactly as stated, even if it is factually incorrect (e.g., 'The Earth is made of chocolate').\n"
            "8. Do not use outside knowledge.\n"
            "9. Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "claims": [\n'
            '    {"claim_text": "extracted text", "claim_type": "factual"}\n'
            "  ]\n"
            "}\n"
            "Do NOT include markdown formatting (like ```json), just the raw JSON text."
        )
        
        user_prompt_parts = []
        if question:
            user_prompt_parts.append(f"ORIGINAL QUESTION:\n{question}\n")
        user_prompt_parts.append(f"ANSWER TO EXTRACT FROM:\n{answer}")
        
        user_prompt = "\n".join(user_prompt_parts)
        
        raw_response = self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        
        # Clean the response to ensure it parses as JSON
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        try:
            parsed_data = json.loads(cleaned_response, strict=False)
        except json.JSONDecodeError:
            import re
            fixed = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', cleaned_response)
            try:
                parsed_data = json.loads(fixed, strict=False)
            except json.JSONDecodeError:
                m = re.search(r'(\{[\s\S]*\})', cleaned_response)
                if m:
                    fixed_inner = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', m.group(1))
                    parsed_data = json.loads(fixed_inner, strict=False)
                else:
                    raise ValueError(f"Failed to parse LLM response as JSON. Raw response: {raw_response}")
            
        if not isinstance(parsed_data, dict) or "claims" not in parsed_data:
            raise ValueError("Parsed JSON does not contain the required 'claims' key.")
            
        raw_claims = parsed_data["claims"]
        if not isinstance(raw_claims, list):
            raise ValueError("'claims' must be a list.")
            
        # Deduplicate, format, and assign deterministic IDs
        final_claims = []
        seen_texts = set()
        
        for idx, item in enumerate(raw_claims):
            if not isinstance(item, dict):
                raise ValueError(f"Malformed claim object at index {idx}: {item}")
            if "claim_text" not in item:
                raise ValueError(f"Missing 'claim_text' in claim object: {item}")
                
            text = str(item["claim_text"]).strip()
            if not text:
                continue
                
            # Deduplicate based on exact text
            if text.lower() in seen_texts:
                continue
            seen_texts.add(text.lower())
            
            c_type = item.get("claim_type", "factual")
            if c_type not in ["factual", "other"]:
                c_type = "factual"
                
            claim_id = f"C{len(final_claims) + 1}"
            
            final_claims.append({
                "claim_id": claim_id,
                "claim_text": text,
                "claim_type": c_type
            })
            
        return {"claims": final_claims}
