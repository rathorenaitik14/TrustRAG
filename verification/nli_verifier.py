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
        model_name: str = "cross-encoder/nli-deberta-v3-small", 
        entailment_threshold: float = 0.5, 
        contradiction_threshold: float = 0.5
    ):
        super().__init__(entailment_threshold, contradiction_threshold)
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
        contradiction_score = float(probs[0])
        entailment_score = float(probs[1])
        neutral_score = float(probs[2])

        # Evaluate logic for this specific evidence chunk
        if contradiction_score > entailment_score and contradiction_score >= self.contradiction_threshold:
            status = "CONTRADICTED"
            score = contradiction_score
        elif entailment_score >= self.entailment_threshold:
            status = "SUPPORTED"
            score = entailment_score
        else:
            status = "UNSUPPORTED"
            score = max(entailment_score, contradiction_score, neutral_score)

        return {
            "status": status,
            "score": score,
            "entailment_score": entailment_score,
            "contradiction_score": contradiction_score,
            "neutral_score": neutral_score
        }
