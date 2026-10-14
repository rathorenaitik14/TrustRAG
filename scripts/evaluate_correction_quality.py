"""
Evaluate Correction Quality (Stage 5B)
======================================
Addresses the "Correction = Removal" problem in prior evaluations.
Evaluates corrections on the 1-5 quality scale:
  1: Claim removed with no replacement (information loss)
  2: Claim heavily modified, loses useful information
  3: Claim appropriately qualified (e.g. "According to the document...")
  4: Claim accurately corrected using evidence
  5: Claim corrected AND explanation/reasoning provided

Outputs:
  - Table of scores and qualitative examples
  - Summary metrics: Mean Quality Score, % Information Preserving, % Removal Only
  - Saves report to results/correction_quality_report.json
"""
import sys
import os
import json
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.correction_quality import CorrectionQualityEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Representative test corrections reflecting real RAG scenarios
SAMPLE_CORRECTIONS = [
    {
        "id": "case_1_removal",
        "original_claim": "The dataset contains 50,000 images collected from ImageNet in 2024.",
        "corrected_claim": "[removed]",
        "evidence": "The benchmark uses 10,000 synthetic images generated in 2023.",
        "verdict": "CONTRADICTED",
        "description": "Baseline: Simple removal of hallucinated claim"
    },
    {
        "id": "case_2_truncated",
        "original_claim": "Transformer architectures rely entirely on recurrent memory cells for sequence processing.",
        "corrected_claim": "Architecture uses cells.",
        "evidence": "Transformers abandon recurrent architectures entirely, relying exclusively on multi-head self-attention mechanisms.",
        "verdict": "CONTRADICTED",
        "description": "Heavily truncated edit with information loss"
    },
    {
        "id": "case_3_qualified",
        "original_claim": "Solar irradiance increased by 15% across Northern Europe throughout 2022.",
        "corrected_claim": "According to the document, solar irradiance levels were recorded across Northern Europe in 2022.",
        "evidence": "According to the study, solar irradiance was monitored across Northern European meteorological stations in 2022 without significant percentage deviations noted.",
        "verdict": "UNSUPPORTED",
        "description": "Hedged/Qualified statement preserving contextual truth"
    },
    {
        "id": "case_4_accurate_correction",
        "original_claim": "BERT was introduced by DeepMind researchers in December 2019.",
        "corrected_claim": "According to the document, BERT was introduced by Google AI Language researchers in October 2018.",
        "evidence": "BERT (Bidirectional Encoder Representations from Transformers) was introduced by researchers at Google AI Language in October 2018.",
        "verdict": "CONTRADICTED",
        "description": "Accurately corrected claim using explicit document evidence"
    },
    {
        "id": "case_5_explained_correction",
        "original_claim": "MiniCheck achieves 0.95 AUC using a 100-layer recurrent neural network.",
        "corrected_claim": "MiniCheck achieves superior fact-checking performance using a DeBERTa-v3-Large cross-encoder because self-attention captures token-level alignment between premise and hypothesis.",
        "evidence": "MiniCheck fine-tunes DeBERTa-v3-Large to detect factual discrepancies between source documents and LLM-generated claims, leveraging dense cross-attention.",
        "verdict": "CONTRADICTED",
        "description": "Correction accompanied by factual explanation and rationale"
    }
]

def main():
    evaluator = CorrectionQualityEvaluator(use_llm=True)
    logger.info("Evaluating sample corrections across quality spectrum...")

    detailed_results = []
    for item in SAMPLE_CORRECTIONS:
        res = evaluator.score_correction(
            original_claim=item["original_claim"],
            corrected_claim=item["corrected_claim"],
            evidence=item["evidence"],
            verdict=item["verdict"]
        )
        detailed_results.append({
            "id": item["id"],
            "description": item["description"],
            "original_claim": item["original_claim"],
            "corrected_claim": item["corrected_claim"],
            "score": res["score"],
            "method": res["method"],
            "rationale": res["rationale"]
        })

    summary = evaluator.score_batch(SAMPLE_CORRECTIONS)
    summary["cases"] = detailed_results

    print("\n" + "="*80)
    print("           CORRECTION QUALITY EVALUATION (STAGE 5B)")
    print("="*80)
    print(f"{'Case ID':<22} | {'Score':<5} | {'Method':<10} | {'Description'}")
    print("-" * 80)
    for res in detailed_results:
        print(f"{res['id']:<22} | {res['score']:<5} | {res['method']:<10} | {res['description']}")
    print("=" * 80)
    print(f"Mean Quality Score:             {summary['mean_score']} / 5.0")
    print(f"% Information-Preserving (>=3): {summary['pct_information_preserving']}%")
    print(f"% Well-Corrected (>=4):         {summary['pct_well_corrected']}%")
    print(f"% Removal Only (Score 1):       {summary['pct_removal_only']}%")
    print("=" * 80 + "\n")

    os.makedirs("results", exist_ok=True)
    output_path = "results/correction_quality_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved correction quality evaluation report to {output_path}")

if __name__ == "__main__":
    main()
