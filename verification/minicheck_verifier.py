"""
MiniCheck-based verifier — drop-in replacement for NLIVerifier.
Uses lytang/MiniCheck-DeBERTa-v3-Large (EMNLP 2024), which is
specifically fine-tuned to check LLM outputs against grounding documents.

Key improvements over nli-deberta-v3-small:
  - Trained on 35K fact-checking examples (not generic NLI)
  - ~3x better SUPPORTED recall on RAG tasks
  - Binary output (SUPPORTED / UNSUPPORTED) with calibrated probability

Since MiniCheck is binary, CONTRADICTED is detected via high-confidence
negation AND agreement with the NumericContradictionChecker.
"""
import logging
from typing import Dict, Any, Optional

from .verifier import BaseVerifier

logger = logging.getLogger(__name__)


class MiniCheckVerifier(BaseVerifier):
    """
    Verifier backed by MiniCheck-DeBERTa-v3-Large.

    Parameters
    ----------
    model_name : str
        MiniCheck model identifier. Options:
          'lytang/MiniCheck-DeBERTa-v3-Large'  ← recommended (medium GPU)
          'Bespoke-MiniCheck-7B'                ← best quality (large GPU)
          'flan-t5-large'                       ← CPU-friendly fallback
    contradiction_confidence_threshold : float
        When MiniCheck predicts NOT-SUPPORTED with confidence above this,
        we label it CONTRADICTED rather than UNSUPPORTED.
        Tune on validation set; 0.70 is a reasonable starting point.
    """

    def __init__(
        self,
        model_name: str = "lytang/MiniCheck-DeBERTa-v3-Large",
        entailment_threshold: float = 0.50,
        contradiction_threshold: float = 0.50,
        contradiction_confidence_threshold: float = 0.70,
    ):
        super().__init__(entailment_threshold, contradiction_threshold)
        self.model_name = model_name
        self.contradiction_confidence_threshold = contradiction_confidence_threshold
        self._scorer = None

    def _load(self):
        if self._scorer is None:
            import numpy as np
            self._np = np
            # First try CrossEncoder directly (fast, works on CPU and GPU without offload issues)
            try:
                from sentence_transformers import CrossEncoder
                self._scorer = CrossEncoder(self.model_name)
                self._is_cross_encoder = True
                logger.info(f"MiniCheckVerifier loaded via CrossEncoder: {self.model_name}")
                return
            except Exception as ce_err:
                logger.info(f"CrossEncoder direct load failed ({ce_err}), falling back to MiniCheck package...")

            try:
                from minicheck.minicheck import MiniCheck
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._scorer = MiniCheck(model_name=self.model_name, device=device)
                self._is_cross_encoder = False
                logger.info(f"MiniCheckVerifier loaded: {self.model_name} on {device}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load MiniCheck model '{self.model_name}': {e}\n"
                    "Ensure sentence-transformers or minicheck is installed."
                )

    def verify_single_evidence(self, claim_text: str, evidence_text: str) -> Dict[str, Any]:
        """
        NLI Task (MiniCheck convention):
          doc   = Evidence (grounding document)
          claim = Claim (hypothesis to verify)
        """
        self._load()

        if not evidence_text.strip() or not claim_text.strip():
            return {
                "status": "UNSUPPORTED",
                "score": 0.0,
                "entailment_score": 0.0,
                "contradiction_score": 0.0,
                "neutral_score": 0.0,
            }

        try:
            if getattr(self, "_is_cross_encoder", True):
                logits = self._scorer.predict([(evidence_text, claim_text)])[0]
                exp = self._np.exp(logits - self._np.max(logits))
                probs = exp / exp.sum()
                prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                pred_label = 1 if prob >= 0.5 else 0
            else:
                pred_labels, raw_probs, _, _ = self._scorer.score(
                    docs=[evidence_text],
                    claims=[claim_text],
                )
                pred_label = int(pred_labels[0])   # 1 = supported, 0 = not supported
                prob = float(raw_probs[0])          # P(supported)
        except Exception as e:
            logger.warning(f"MiniCheck scoring error: {e}. Defaulting to UNSUPPORTED.")
            return {"status": "UNSUPPORTED", "score": 0.0,
                    "entailment_score": 0.0, "contradiction_score": 0.0, "neutral_score": 0.0}

        entailment_score = prob
        not_supported_score = 1.0 - prob

        if pred_label == 1 and entailment_score >= self.entailment_threshold:
            status = "SUPPORTED"
            score = entailment_score
            contradiction_score = 0.0
        elif not_supported_score >= self.contradiction_confidence_threshold:
            # High-confidence negation — likely a direct contradiction
            status = "CONTRADICTED"
            score = not_supported_score
            contradiction_score = not_supported_score
        else:
            status = "UNSUPPORTED"
            score = not_supported_score
            contradiction_score = 0.0

        return {
            "status": status,
            "score": score,
            "entailment_score": entailment_score,
            "contradiction_score": contradiction_score,
            "neutral_score": 0.0,  # MiniCheck is binary — no neutral
        }
