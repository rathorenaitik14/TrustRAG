"""
HaluEval-QA Dataset Loader
===========================
Loads the HaluEval QA split (EMNLP 2023) from Hugging Face and converts it
into the UnifiedBenchmarkCase format.

Dataset: pminervini/HaluEval  (configuration: "qa")
Paper:   "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs"
         Li et al., EMNLP 2023.

Each HaluEval-QA record has:
  - question        : the user question
  - right_answer    : a factually correct answer
  - hallucinated_answer : an answer with at least one hallucination
  - knowledge       : the grounding knowledge snippet

We create TWO cases per record:
  1. right_answer    → SUPPORTED
  2. hallucinated_answer → UNSUPPORTED  (HaluEval doesn't distinguish
                                          UNSUPPORTED vs CONTRADICTED,
                                          so everything bad → UNSUPPORTED)

Usage:
    loader = HaluEvalLoader()
    cases  = loader.load(n_samples=300, seed=42)
"""
import random
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class HaluEvalLoader:
    """
    Loads HaluEval-QA and converts to UnifiedBenchmarkCase format.
    """

    DATASET_NAME = "pminervini/HaluEval"
    CONFIG_NAME = "qa"

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        config_name: str = CONFIG_NAME,
    ):
        self.dataset_name = dataset_name
        self.config_name = config_name

    def load(
        self,
        n_samples: int = 300,
        seed: int = 42,
        split: str = "data",
        balanced: bool = True,
    ) -> List[dict]:
        """
        Load n_samples cases from HaluEval-QA.

        Parameters
        ----------
        n_samples : Total cases to return. We create 2 cases per record
                    (one SUPPORTED, one UNSUPPORTED), so we need n_samples/2
                    raw records.
        seed      : Random seed.
        split     : Dataset split name (HaluEval uses "data").
        balanced  : If True, equal SUPPORTED/UNSUPPORTED split (default).

        Returns
        -------
        List of UnifiedBenchmarkCase dicts.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required to load HaluEval.\n"
                "Install it with: pip install datasets"
            )

        logger.info(f"Loading HaluEval ({self.config_name}) from '{self.dataset_name}' ...")
        try:
            ds = load_dataset(self.dataset_name, self.config_name, split=split)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HaluEval dataset.\n"
                f"Try: from datasets import load_dataset; "
                f"load_dataset('{self.dataset_name}', '{self.config_name}')\n"
                f"Original error: {e}"
            )

        records = list(ds)
        logger.info(f"HaluEval: {len(records)} raw records loaded.")

        rng = random.Random(seed)
        # We produce 2 cases per record: take ceil(n_samples/2) records
        n_records = min((n_samples + 1) // 2, len(records))
        selected = rng.sample(records, n_records)

        cases = []
        for idx, record in enumerate(selected):
            # Case A: correct answer → SUPPORTED
            cases.append(self._convert(record, idx * 2, hallucinated=False))
            # Case B: hallucinated answer → UNSUPPORTED
            cases.append(self._convert(record, idx * 2 + 1, hallucinated=True))

        # Trim to exactly n_samples and re-shuffle
        cases = cases[:n_samples]
        rng.shuffle(cases)

        logger.info(f"Sampled {len(cases)} HaluEval-QA cases.")
        return cases

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _convert(self, record: dict, idx: int, hallucinated: bool) -> dict:
        """Convert one HaluEval record to UnifiedBenchmarkCase."""
        question = str(record.get("question", ""))
        knowledge = str(record.get("knowledge", ""))
        right_answer = str(record.get("right_answer", ""))
        hal_answer = str(record.get("hallucinated_answer", ""))

        label = "UNSUPPORTED" if hallucinated else "SUPPORTED"
        answer = hal_answer if hallucinated else right_answer
        expected_action = "REMOVED_OR_QUALIFIED" if hallucinated else "PRESERVED"
        case_id = f"halueval-{'h' if hallucinated else 's'}-{idx:05d}"

        return {
            # Unified fields
            "case_id": case_id,
            "source_dataset": "halueval",
            "task_type": "qa",
            "question": question,
            "context": knowledge,
            "answer": answer,
            "ground_truth_label": label,
            "difficulty": "medium" if hallucinated else "easy",
            "notes": f"HaluEval-QA {'hallucinated' if hallucinated else 'correct'} answer",

            # BenchmarkRunner compatibility fields
            "document_id": f"halueval-doc-{idx}",
            "document_name": f"HaluEval knowledge snippet {idx}",
            "grounded_answer": right_answer,
            "injected_answer": answer,
            "original_claims": [right_answer],
            "injected_claim": hal_answer if hallucinated else "",
            "evidence_reference": knowledge[:300],
            "expected_correction_action": expected_action,
        }
