class HallucinationInjector:
    """
    Deterministic hallucination injector for benchmark datasets.
    """
    def append_claim(self, grounded_answer: str, claim_to_inject: str) -> str:
        if not grounded_answer:
            return claim_to_inject
        
        grounded_answer = grounded_answer.strip()
        claim_to_inject = claim_to_inject.strip()
        
        if not grounded_answer.endswith("."):
            grounded_answer += "."
            
        return f"{grounded_answer} {claim_to_inject}"
        
    def replace_phrase(self, grounded_answer: str, original_phrase: str, new_phrase: str) -> str:
        if original_phrase not in grounded_answer:
            raise ValueError(f"Phrase '{original_phrase}' not found in grounded answer.")
        return grounded_answer.replace(original_phrase, new_phrase)
        
    def insert_claim(self, grounded_answer: str, claim_to_inject: str, insert_after: str) -> str:
        if insert_after not in grounded_answer:
            raise ValueError(f"Insert anchor '{insert_after}' not found in grounded answer.")
            
        parts = grounded_answer.split(insert_after, 1)
        return f"{parts[0]}{insert_after} {claim_to_inject.strip()} {parts[1]}".strip()
