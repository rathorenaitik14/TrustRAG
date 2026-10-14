"""
RAGTruth Dataset Loader
=======================
Loads the RAGTruth hallucination corpus (ACL 2024) from Hugging Face and
converts it into the UnifiedBenchmarkCase format.

Dataset: wandb/RAGTruth-processed  (≈18K responses, word-level annotations)
Paper:   "RAGTruth: A Hallucination Corpus for Developing Trustworthy RAL Models"
         Zhang et al., ACL 2024.

Label mapping:
  - response has NO hallucinated spans  → SUPPORTED
  - response has hallucinated spans that add unsupported info → UNSUPPORTED
  - response has hallucinated spans that directly contradict the passage → CONTRADICTED

Usage:
    loader = RAGTruthLoader()
    cases  = loader.load(n_samples=500, seed=42, split="test")
"""
import random
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class RAGTruthLoader:
    """
    Loads and converts RAGTruth into UnifiedBenchmarkCase format.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset identifier. Default is the processed variant.
    """

    DATASET_NAME = "wandb/RAGTruth-processed"
    LABEL_MAP = {
        # hallucination_type values in RAGTruth-processed
        "None": "SUPPORTED",
        "Evident Conflict": "CONTRADICTED",
        "Subtle Conflict": "CONTRADICTED",
        "Evident Baseless Info": "UNSUPPORTED",
        "Subtle Baseless Info": "UNSUPPORTED",
    }

    def __init__(self, dataset_name: str = DATASET_NAME):
        self.dataset_name = dataset_name

    def load(
        self,
        n_samples: int = 500,
        seed: int = 42,
        split: str = "train",           # RAGTruth-processed only has "train"
        task_filter: Optional[str] = None,  # "QA" | "Summary" | "Data2Text"
        stratify: bool = True,
    ) -> List[dict]:
        """
        Load n_samples cases from RAGTruth.

        Parameters
        ----------
        n_samples  : Target total cases (stratified across SUPPORTED/UNSUPPORTED/CONTRADICTED).
        seed       : Random seed for reproducibility.
        split      : Dataset split (default "train" — only split in the processed version).
        task_filter: If set, restrict to a specific task type.
        stratify   : If True, balance labels (33% each). If False, use natural distribution.

        Returns
        -------
        List of UnifiedBenchmarkCase dicts.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required to load RAGTruth.\n"
                "Install it with: pip install datasets"
            )

        logger.info(f"Loading RAGTruth from '{self.dataset_name}' split='{split}' ...")
        try:
            ds = load_dataset(self.dataset_name, split=split)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load RAGTruth dataset '{self.dataset_name}'.\n"
                f"Ensure you have an internet connection and the dataset exists.\n"
                f"Original error: {e}"
            )

        records = list(ds)
        logger.info(f"RAGTruth: {len(records)} total records loaded.")

        # Optional task filter
        if task_filter:
            records = [
                r for r in records
                if str(r.get("task_type", "")).lower() == task_filter.lower()
            ]
            logger.info(f"After task filter '{task_filter}': {len(records)} records.")

        if stratify:
            cases = self._stratified_sample(records, n_samples, seed)
        else:
            rng = random.Random(seed)
            cases = rng.sample(records, min(n_samples, len(records)))

        logger.info(f"Sampled {len(cases)} RAGTruth cases.")
        return [self._convert(r, idx) for idx, r in enumerate(cases)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stratified_sample(self, records: list, n: int, seed: int) -> list:
        """Sample n records, aiming for ~equal distribution of labels."""
        rng = random.Random(seed)
        buckets: dict = {"SUPPORTED": [], "UNSUPPORTED": [], "CONTRADICTED": []}

        for r in records:
            label = self._infer_label(r)
            if label in buckets:
                buckets[label].append(r)

        per_class = n // 3
        remainder = n % 3
        sampled = []
        labels_in_order = ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]

        for i, label in enumerate(labels_in_order):
            take = per_class + (1 if i < remainder else 0)
            bucket = buckets[label]
            rng.shuffle(bucket)
            sampled.extend(bucket[:take])

        rng.shuffle(sampled)
        return sampled

    def _infer_label(self, record: dict) -> str:
        """
        Map RAGTruth hallucination_type → our 3-class label.
        If the record has a list of hallucination spans, pick the most
        severe label.
        """
        htype = record.get("hallucination_type", "None")
        # Some versions store a list of types
        if isinstance(htype, list):
            types = htype
        else:
            types = [str(htype)]

        # Priority: CONTRADICTED > UNSUPPORTED > SUPPORTED
        if any(t in ("Evident Conflict", "Subtle Conflict") for t in types):
            return "CONTRADICTED"
        if any(t in ("Evident Baseless Info", "Subtle Baseless Info") for t in types):
            return "UNSUPPORTED"
        return "SUPPORTED"

    def _infer_difficulty(self, record: dict) -> str:
        htype = str(record.get("hallucination_type", "None"))
        if "Subtle" in htype:
            return "hard"
        if "Evident" in htype:
            return "medium"
        return "easy"

    def _expected_correction(self, label: str) -> str:
        return {
            "SUPPORTED": "PRESERVED",
            "UNSUPPORTED": "REMOVED_OR_QUALIFIED",
            "CONTRADICTED": "CORRECTED",
        }.get(label, "PRESERVED")

    def _convert(self, record: dict, idx: int) -> dict:
        """Convert a single RAGTruth record to UnifiedBenchmarkCase."""
        label = self._infer_label(record)
        difficulty = self._infer_difficulty(record)

        # RAGTruth-processed field names (may vary across versions)
        question = str(record.get("question", record.get("source", f"Query {idx}")))
        context = str(record.get("passage", record.get("context", record.get("source_doc", ""))))
        answer = str(record.get("response", record.get("answer", "")))
        task_type = str(record.get("task_type", "qa")).lower()

        # Hallucinated span text (if available) — used as the "injected claim"
        hal_span = record.get("hallucinated_span", record.get("hallucination_span", ""))
        if isinstance(hal_span, list):
            hal_span = " ".join(str(s) for s in hal_span)
        hal_span = str(hal_span).strip()

        case_id = f"ragtruth-{idx:05d}"

        return {
            # Unified fields
            "case_id": case_id,
            "source_dataset": "ragtruth",
            "task_type": task_type,
            "question": question,
            "context": context,
            "answer": answer,
            "ground_truth_label": label,
            "difficulty": difficulty,
            "notes": f"RAGTruth hallucination_type={record.get('hallucination_type', 'None')}",

            # BenchmarkRunner compatibility fields
            "document_id": f"ragtruth-doc-{idx}",
            "document_name": f"RAGTruth passage {idx}",
            "grounded_answer": answer,
            "injected_answer": answer,
            "original_claims": [answer],
            "injected_claim": hal_span,
            "evidence_reference": context[:300],  # first 300 chars as reference
            "expected_correction_action": self._expected_correction(label),
        }
