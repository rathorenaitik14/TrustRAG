from typing import Dict, Any
import numpy as np
from sentence_transformers import CrossEncoder
from .verifier import BaseVerifier

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class NLIVerifier(BaseVerifier):
    """
    NLI-based Verifier using a pretrained CrossEncoder.
    Default model: cross-encoder/nli-deberta-v3-small.
    The model outputs logits for [contradiction, entailment, neutral].
    """
    def __init__(
        self, 
        model_name: str = "lytang/MiniCheck-DeBERTa-v3-Large", 
        entailment_threshold: float = 0.5, 
        contradiction_threshold: float = 0.5,
        use_numeric_checker: bool = True
    ):
        super().__init__(entailment_threshold, contradiction_threshold, use_numeric_checker)
        self.model_name = model_name
        # We use CrossEncoder which returns logits for pairs
        self.model = CrossEncoder(model_name)

    def verify_single_evidence(self, claim_text: str, evidence_text: str) -> Dict[str, Any]:
        """
        NLI Task:
        Premise = Evidence
        Hypothesis = Claim
        """
        if not evidence_text.strip() or not claim_text.strip():
            return {"status": "UNSUPPORTED", "score": 0.0}

        # Predict logits
        # Important: For NLI, it's (Premise, Hypothesis)
        logits = self.model.predict([(evidence_text, claim_text)])[0]
        
        # cross-encoder/nli-deberta-v3-small mapping:
        # 0: contradiction, 1: entailment, 2: neutral
        probs = softmax(logits)
        if len(probs) >= 3:
            contradiction_score = float(probs[0])
            entailment_score = float(probs[1])
            neutral_score = float(probs[2])
            if contradiction_score > entailment_score and contradiction_score >= self.contradiction_threshold:
                status = "CONTRADICTED"
                score = contradiction_score
            elif entailment_score >= self.entailment_threshold:
                status = "SUPPORTED"
                score = entailment_score
            else:
                status = "UNSUPPORTED"
                score = max(entailment_score, contradiction_score, neutral_score)
        else:
            # Binary fact-checking model (e.g. MiniCheck): [0: not supported, 1: supported]
            not_supported_score = float(probs[0])
            entailment_score = float(probs[1])
            neutral_score = 0.0
            contradiction_score = not_supported_score if not_supported_score >= 0.70 else 0.0

            if entailment_score >= self.entailment_threshold:
                status = "SUPPORTED"
                score = entailment_score
            elif not_supported_score >= 0.70:
                status = "CONTRADICTED"
                score = not_supported_score
            else:
                status = "UNSUPPORTED"
                score = not_supported_score

        return {
            "status": status,
            "score": score,
            "entailment_score": entailment_score,
            "contradiction_score": contradiction_score,
            "neutral_score": neutral_score
        }
