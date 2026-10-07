import json
from typing import List, Optional
from claims.models import Claim, ClaimEvidenceResult
from verification.models import ClaimVerificationResult
from generation.llm_client import LLMClient, get_llm_client
from .models import CorrectionResult, CorrectionChange
from .corrector import BaseCorrector

class GeminiCorrector(BaseCorrector):
    """
    Evidence-grounded corrector using Gemini.
    """
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_llm_client()

    def correct(
        self,
        original_answer: str,
        claims: List[Claim],
        evidence_results: List[ClaimEvidenceResult],
        verification_results: List[ClaimVerificationResult]
    ) -> CorrectionResult:
        
        system_prompt = (
            "You are correcting an answer using only the supplied evidence.\n\n"
            "You must not introduce facts from your own knowledge.\n"
            "Preserve claims marked SUPPORTED.\n"
            "For CONTRADICTED claims, replace the claim only with information supported by the provided evidence.\n"
            "For UNSUPPORTED claims, do not guess or invent a replacement. Remove the unsupported assertion or explicitly state that the provided evidence does not establish it.\n"
            "Do not invent citations.\n"
            "Do not invent evidence.\n"
            "Do not change supported factual claims unnecessarily.\n"
            "Return a concise final answer.\n\n"
            "Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "original_answer": "...",\n'
            '  "corrected_answer": "...",\n'
            '  "changes": [\n'
            '    {\n'
            '      "claim_id": "C1",\n'
            '      "status": "SUPPORTED",\n'
            '      "action": "PRESERVED"\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "Allowed actions for each claim are: PRESERVED, CORRECTED, REMOVED_OR_QUALIFIED.\n"
            "Do NOT include markdown formatting (like ```json), just the raw JSON text."
        )

        # Build User Prompt
        user_prompt_parts = []
        user_prompt_parts.append(f"ORIGINAL ANSWER:\n{original_answer}\n")
        user_prompt_parts.append("CLAIMS:\n")

        # Map evidence texts for quick lookup
        evidence_text_map = {}
        for er in evidence_results:
            for ev in er.get("evidence", []):
                evidence_text_map[ev["evidence_id"]] = ev.get("text", "")

        for vr in verification_results:
            claim_id = vr["claim_id"]
            user_prompt_parts.append(f"{claim_id}")
            user_prompt_parts.append(f"Claim: {vr['claim_text']}")
            user_prompt_parts.append(f"Status: {vr['status']}")
            
            # Find evidence details
            ev_texts = []
            for ev_det in vr.get("evidence_details", []):
                ev_id = ev_det["evidence_id"]
                ev_texts.append(f"{ev_id}: {evidence_text_map.get(ev_id, 'No text')}")
            
            if ev_texts:
                user_prompt_parts.append("Evidence:")
                user_prompt_parts.extend(ev_texts)
            else:
                user_prompt_parts.append("Evidence: []")
            user_prompt_parts.append("") # empty line

        user_prompt = "\n".join(user_prompt_parts)

        raw_response = self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt)

        # Clean JSON
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        try:
            parsed_data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}. Raw response: {raw_response}")

        # Validate structure
        if "original_answer" not in parsed_data or "corrected_answer" not in parsed_data or "changes" not in parsed_data:
            raise ValueError("Parsed JSON is missing required fields.")

        # Ensure valid actions
        valid_actions = {"PRESERVED", "CORRECTED", "REMOVED_OR_QUALIFIED"}
        for c in parsed_data.get("changes", []):
            if c.get("action") not in valid_actions:
                c["action"] = "REMOVED_OR_QUALIFIED" # fallback if LLM hallucinates action

        return {
            "original_answer": str(parsed_data["original_answer"]),
            "corrected_answer": str(parsed_data["corrected_answer"]),
            "changes": parsed_data["changes"]
        }
