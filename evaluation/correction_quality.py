"""
Correction Quality Evaluator
==============================
Evaluates whether the corrected answer preserves useful content AND
fixes the hallucination. Addresses the weakness that "removed" currently
counts as full success in the original paper.

Two evaluation modes:
  1. LLM-as-Judge   – uses Gemini to score corrections 1–5
  2. Heuristic      – fast, no-API rule-based scoring

Scoring rubric (1–5):
  1 = Claim removed with no replacement (information loss)
  2 = Claim heavily modified, loses useful information
  3 = Claim appropriately qualified (e.g., "according to the document...")
  4 = Claim accurately corrected with evidence
  5 = Claim corrected AND reasoning/explanation provided

Usage:
    evaluator = CorrectionQualityEvaluator()
    score = evaluator.score_correction(
        original_claim="Data mining was invented at MIT.",
        corrected_claim="[claim removed]",
        evidence="Goal of Data Mining is to provide efficient tools for KDD.",
        verdict="UNSUPPORTED"
    )
"""
import re
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score bands
# ---------------------------------------------------------------------------

SCORE_DESCRIPTIONS = {
    1: "Claim removed with no replacement (information loss)",
    2: "Claim heavily modified, loses useful information",
    3: "Claim appropriately qualified or hedged",
    4: "Claim accurately corrected using evidence",
    5: "Claim corrected with reasoning/explanation provided",
}

# Phrases indicating removal/deletion
_REMOVAL_RE = re.compile(
    r'\[?(removed?|deleted?|omitted?|N/?A|not (supported|verified))\]?',
    re.IGNORECASE
)
# Qualification phrases
_QUALIFICATION_RE = re.compile(
    r'\b(according to|based on|the (document|source|text) (states?|says?|mentions?)|'
    r'as stated|as noted|per the|the evidence (suggests?|indicates?))\b',
    re.IGNORECASE
)
# Explanation phrases
_EXPLANATION_RE = re.compile(
    r'\b(because|therefore|since|this (means?|indicates?|shows?)|'
    r'the reason|specifically|in fact|to clarify)\b',
    re.IGNORECASE
)


class CorrectionQualityEvaluator:
    """
    Evaluates correction quality with optional LLM scoring.
    Always falls back to heuristic scoring if LLM is unavailable.
    """

    SYSTEM_PROMPT = """You are evaluating the quality of a factual correction made by an AI system.
Given the original claim, the corrected version, and the supporting evidence, rate the correction quality on a scale of 1-5:

1 = Claim simply removed or replaced with "[removed]" — information lost
2 = Claim heavily modified, useful information lost
3 = Claim appropriately qualified (e.g., "According to the document...")
4 = Claim accurately corrected using the evidence
5 = Claim corrected AND explanation or reasoning is provided

Output ONLY the integer score (1, 2, 3, 4, or 5). No explanation."""

    def __init__(self, use_llm: bool = True, llm_model: Optional[str] = None):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        self.use_llm = use_llm
        self.llm_model = llm_model or os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
        self._client = None

    def _load_llm(self):
        if self._client is None:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            from google import genai
            from google.genai import types
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY, GEMINI_API_KEY, or LLM_API_KEY not set.")
            self._client = genai.Client(api_key=api_key)
            self._types = types

    def score_correction(
        self,
        original_claim: str,
        corrected_claim: str,
        evidence: str,
        verdict: str,  # "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED"
    ) -> Dict[str, Any]:
        """
        Score a single correction.

        Returns
        -------
        dict with keys: score (1–5), method, rationale
        """
        heuristic_result = self._heuristic_score(original_claim, corrected_claim, verdict)

        if self.use_llm:
            try:
                llm_result = self._llm_score(original_claim, corrected_claim, evidence, verdict)
                return {
                    "score": llm_result["score"],
                    "heuristic_score": heuristic_result["score"],
                    "method": "llm",
                    "rationale": SCORE_DESCRIPTIONS[llm_result["score"]],
                    "llm_raw": llm_result.get("raw", ""),
                }
            except Exception as e:
                logger.warning(f"LLM scoring failed ({e}), using heuristic.")

        return {
            "score": heuristic_result["score"],
            "method": "heuristic",
            "rationale": heuristic_result["rationale"],
        }

    def score_batch(
        self,
        corrections: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Score a batch of corrections and return aggregate statistics.

        Each item in `corrections` must have keys:
          original_claim, corrected_claim, evidence, verdict
        """
        scores = []
        for item in corrections:
            result = self.score_correction(
                original_claim=item.get("original_claim", ""),
                corrected_claim=item.get("corrected_claim", ""),
                evidence=item.get("evidence", ""),
                verdict=item.get("verdict", "UNSUPPORTED"),
            )
            scores.append(result["score"])

        if not scores:
            return {"error": "No corrections to evaluate."}

        return {
            "n": len(scores),
            "mean_score": round(sum(scores) / len(scores), 3),
            "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
            "pct_information_preserving": round(
                sum(1 for s in scores if s >= 3) / len(scores) * 100, 1
            ),
            "pct_well_corrected": round(
                sum(1 for s in scores if s >= 4) / len(scores) * 100, 1
            ),
            "pct_removal_only": round(
                sum(1 for s in scores if s == 1) / len(scores) * 100, 1
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _heuristic_score(self, original: str, corrected: str, verdict: str) -> Dict[str, Any]:
        """Fast rule-based quality estimation."""
        # Score 1: Claim removed entirely
        if _REMOVAL_RE.search(corrected) or not corrected.strip() or corrected == "[removed]":
            return {"score": 1, "rationale": SCORE_DESCRIPTIONS[1]}

        # Score 2: Very short and doesn't contain content from original (truncated)
        if len(corrected) < 0.3 * len(original) and len(corrected) < 50:
            return {"score": 2, "rationale": SCORE_DESCRIPTIONS[2]}

        # Score 5: Has explanation/reasoning
        if _EXPLANATION_RE.search(corrected):
            return {"score": 5, "rationale": SCORE_DESCRIPTIONS[5]}

        # Score 4: Has qualification AND actual correction
        if _QUALIFICATION_RE.search(corrected):
            return {"score": 4, "rationale": SCORE_DESCRIPTIONS[4]}

        # Score 3: Modified but no clear qualification
        if corrected.strip() != original.strip():
            return {"score": 3, "rationale": SCORE_DESCRIPTIONS[3]}

        # Score 1: Unchanged (correction did nothing)
        return {"score": 1, "rationale": "Correction appears unchanged from original."}

    def _llm_score(
        self, original: str, corrected: str, evidence: str, verdict: str
    ) -> Dict[str, Any]:
        """Score using Gemini."""
        self._load_llm()

        user_prompt = (
            f"ORIGINAL CLAIM: {original}\n\n"
            f"CORRECTED VERSION: {corrected}\n\n"
            f"EVIDENCE: {evidence[:500]}\n\n"
            f"SYSTEM VERDICT: {verdict}\n\n"
            "Score (1-5):"
        )

        response = self._client.models.generate_content(
            model=self.llm_model,
            contents=user_prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=64,
            )
        )
        raw = ""
        if response and getattr(response, "text", None):
            raw = response.text.strip()
        elif response and getattr(response, "candidates", None) and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    raw += part.text
            raw = raw.strip()

        match = re.search(r'[1-5]', raw)
        score = int(match.group(0)) if match else 3

        return {"score": score, "raw": raw}
