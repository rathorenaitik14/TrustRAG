"""
Baseline verifier implementations for head-to-head comparison with TrustRAG.

Four baselines:
  1. PlainNLIBaseline      – NLI on claim vs raw context, no retrieval
  2. LLMJudgeBaseline      – Gemini prompt-based judge (uses existing google-genai dep)
  3. RAGASFaithfulness     – RAGAS faithfulness metric
  4. MiniCheckBaseline     – MiniCheck-DeBERTa-v3-Large (strong NLI baseline)

All implement a common interface: verify(question, context, answer) → dict
"""
import logging
import re
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared result schema
# ---------------------------------------------------------------------------

def _make_result(label: str, score: float, raw: Optional[dict] = None) -> dict:
    """Normalise all baseline outputs to the same dict shape."""
    return {
        "label": label,            # SUPPORTED | UNSUPPORTED | CONTRADICTED
        "score": round(score, 4),  # confidence [0, 1]
        "raw": raw or {},
    }


# ---------------------------------------------------------------------------
# Baseline 1 – Plain NLI (no retrieval)
# ---------------------------------------------------------------------------

class PlainNLIBaseline:
    """
    Verify a claim directly against a context string using a cross-encoder NLI
    model.  No FAISS retrieval, no reranker – just NLI.

    This is the weakest but fastest baseline and isolates the NLI component.
    """

    MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                import numpy as np
                self._np = np
                self._model = CrossEncoder(self.model_name)
                logger.info(f"PlainNLI loaded: {self.model_name}")
            except ImportError:
                raise ImportError("pip install sentence-transformers")

    def verify(self, question: str, context: str, answer: str) -> dict:
        """
        Verifies `answer` against `context`.
        NLI convention: premise = context, hypothesis = answer.
        """
        self._load()

        if not context.strip() or not answer.strip():
            return _make_result("UNSUPPORTED", 0.0)

        logits = self._model.predict([(context, answer)])[0]
        # cross-encoder/nli-deberta-v3-small: [contradiction, entailment, neutral]
        exp = self._np.exp(logits - self._np.max(logits))
        probs = exp / exp.sum()

        contra, entail, neutral = float(probs[0]), float(probs[1]), float(probs[2])

        if contra > entail and contra >= 0.5:
            label, score = "CONTRADICTED", contra
        elif entail >= 0.5:
            label, score = "SUPPORTED", entail
        else:
            label, score = "UNSUPPORTED", max(neutral, 1 - entail - contra)

        return _make_result(label, score, {
            "entailment": entail,
            "contradiction": contra,
            "neutral": neutral,
        })


# ---------------------------------------------------------------------------
# Baseline 2 – LLM-as-Judge (Gemini)
# ---------------------------------------------------------------------------

class LLMJudgeBaseline:
    """
    Uses Gemini (google-genai) to judge whether an answer is SUPPORTED,
    UNSUPPORTED, or CONTRADICTED by the provided context.

    Requires GOOGLE_API_KEY in environment (same as your existing .env).
    """

    SYSTEM_PROMPT = (
        "You are a precise fact-checking assistant. "
        "You will be given a CONTEXT and a CLAIM. "
        "Respond with exactly one of these three words:\n"
        "  SUPPORTED   – the claim is clearly supported by the context\n"
        "  UNSUPPORTED – the claim is not mentioned or cannot be verified from the context\n"
        "  CONTRADICTED – the claim directly contradicts information in the context\n"
        "Do NOT add any explanation. Output only the single word."
    )

    LABEL_RE = re.compile(r'\b(SUPPORTED|UNSUPPORTED|CONTRADICTED)\b', re.IGNORECASE)

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        import os
        self.model_name = model_name or os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"
        self.temperature = temperature
        self._client = None

    def _load(self):
        if self._client is None:
            try:
                import os
                from google import genai
                from google.genai import types
                api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
                if not api_key:
                    raise EnvironmentError(
                        "GOOGLE_API_KEY, GEMINI_API_KEY, or LLM_API_KEY not set. "
                        "Add it to your .env file."
                    )
                self._client = genai.Client(api_key=api_key)
                self._types = types
                logger.info(f"LLMJudge loaded: {self.model_name}")
            except ImportError:
                raise ImportError("pip install google-genai")

    def verify(self, question: str, context: str, answer: str) -> dict:
        self._load()

        user_prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"CLAIM:\n{answer}\n\n"
            "Your verdict:"
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_output_tokens=10,
                )
            )
            raw_text = response.text.strip().upper()
        except Exception as e:
            logger.warning(f"LLMJudge API error: {e}. Defaulting to UNSUPPORTED.")
            return _make_result("UNSUPPORTED", 0.0, {"error": str(e)})

        match = self.LABEL_RE.search(raw_text)
        label = match.group(1).upper() if match else "UNSUPPORTED"

        return _make_result(label, 1.0 if match else 0.5, {"raw_response": raw_text})


# ---------------------------------------------------------------------------
# Baseline 3 – RAGAS Faithfulness
# ---------------------------------------------------------------------------

class RAGASFaithfulness:
    """
    Uses RAGAS faithfulness metric to score whether an answer is faithful to
    the retrieved context. Returns a score in [0, 1].

    Scores are mapped to labels:
      ≥ 0.85 → SUPPORTED
      ≤ 0.30 → UNSUPPORTED (or CONTRADICTED — RAGAS doesn't distinguish)
      else   → UNSUPPORTED
    """

    SUPPORTED_THRESHOLD = 0.85
    UNSUPPORTED_THRESHOLD = 0.30

    def __init__(self, llm_model: Optional[str] = None):
        import os
        self.llm_model = llm_model or os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
        self._initialized = False

    def _load(self):
        if not self._initialized:
            try:
                import os
                from ragas import evaluate
                from ragas.metrics import faithfulness
                from ragas.llms import LangchainLLMWrapper
                self._evaluate = evaluate
                self._faithfulness = faithfulness
                self._initialized = True
                logger.info("RAGAS faithfulness metric loaded.")
            except ImportError:
                raise ImportError("pip install ragas langchain-google-genai")

    def verify(self, question: str, context: str, answer: str) -> dict:
        """
        Run RAGAS faithfulness on a single (question, context, answer) triple.
        """
        self._load()

        try:
            from datasets import Dataset as HFDataset
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [[context]],
                "ground_truth": [answer],  # RAGAS needs this field even if unused
            }
            hf_ds = HFDataset.from_dict(data)
            result = self._evaluate(hf_ds, metrics=[self._faithfulness])
            score = float(result["faithfulness"])
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {e}. Defaulting score=0.")
            return _make_result("UNSUPPORTED", 0.0, {"error": str(e)})

        if score >= self.SUPPORTED_THRESHOLD:
            label = "SUPPORTED"
        else:
            label = "UNSUPPORTED"

        return _make_result(label, score, {"faithfulness_score": score})


# ---------------------------------------------------------------------------
# Baseline 4 – MiniCheck (Strong NLI)
# ---------------------------------------------------------------------------

class MiniCheckBaseline:
    """
    Uses MiniCheck-DeBERTa-v3-Large — a model specifically fine-tuned for
    fact-checking LLM outputs against grounding documents (EMNLP 2024).

    This is the strongest standalone baseline and a key comparison for TrustRAG.
    Note: MiniCheck only predicts SUPPORTED vs UNSUPPORTED (binary). We return
    CONTRADICTED only when confidence ≥ 0.8 for "unsupported".
    """

    # Fallback to a lighter model if GPU is unavailable
    MODEL_OPTIONS = {
        "large": "Bespoke-MiniCheck-7B",
        "medium": "lytang/MiniCheck-DeBERTa-v3-Large", 
        "small": "flan-t5-large",
    }

    def __init__(self, model_size: str = "medium"):
        if model_size in self.MODEL_OPTIONS:
            self.model_name = self.MODEL_OPTIONS[model_size]
        else:
            self.model_name = model_size
        self._scorer = None
        self._is_cross_encoder = True

    def _load(self):
        if self._scorer is None:
            import numpy as np
            self._np = np
            # Try CrossEncoder directly
            try:
                from sentence_transformers import CrossEncoder
                self._scorer = CrossEncoder(self.model_name)
                self._is_cross_encoder = True
                logger.info(f"MiniCheckBaseline loaded via CrossEncoder: {self.model_name}")
                return
            except Exception as ce_err:
                logger.info(f"CrossEncoder load note ({ce_err}), falling back to MiniCheck package...")

            try:
                from minicheck.minicheck import MiniCheck
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._scorer = MiniCheck(model_name=self.model_name, device=device)
                self._is_cross_encoder = False
                logger.info(f"MiniCheck loaded: {self.model_name}")
            except Exception as e:
                raise ImportError(
                    f"Failed to load MiniCheck model '{self.model_name}': {e}\n"
                    "Ensure sentence-transformers or minicheck is installed."
                )

    def verify(self, question: str, context: str, answer: str) -> dict:
        self._load()

        if not context.strip() or not answer.strip():
            return _make_result("UNSUPPORTED", 0.0)

        try:
            if self._is_cross_encoder:
                logits = self._scorer.predict([(context, answer)])[0]
                exp = self._np.exp(logits - self._np.max(logits))
                probs = exp / exp.sum()
                prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                pred_label = 1 if prob >= 0.5 else 0
            else:
                pred_labels, raw_probs, _, _ = self._scorer.score(
                    docs=[context],
                    claims=[answer]
                )
                pred_label = int(pred_labels[0])  # 1 = supported, 0 = not supported
                prob = float(raw_probs[0])
        except Exception as e:
            logger.warning(f"MiniCheck scoring failed: {e}")
            return _make_result("UNSUPPORTED", 0.0, {"error": str(e)})

        if pred_label == 1:
            label = "SUPPORTED"
            score = prob
        else:
            # High-confidence unsupported → likely CONTRADICTED
            if (1.0 - prob) >= 0.70:
                label = "CONTRADICTED"
            else:
                label = "UNSUPPORTED"
            score = 1.0 - prob

        return _make_result(label, score, {
            "minicheck_label": pred_label,
            "minicheck_prob": prob,
        })
