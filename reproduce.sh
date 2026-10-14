#!/usr/bin/env bash
# =============================================================================
# TrustRAG Reproduction Script
# Reproduces all results from the paper in 4 steps.
# =============================================================================
set -e

echo "============================================================"
echo " TrustRAG — Full Reproduction"
echo "============================================================"

# ── Step 0: Install dependencies ─────────────────────────────────────────────
echo ""
echo "[Step 0] Installing dependencies..."
pip install -r requirements.txt --quiet
# MiniCheck may need source install if pip fails:
# pip install git+https://github.com/Liyan06/MiniCheck.git

# ── Step 1: Head-to-Head Baseline Evaluation (Table 2 in paper) ──────────────
echo ""
echo "[Step 1] Running Head-to-Head Baseline Evaluation on RAGTruth (Table 2)..."
python scripts/evaluate_baselines.py --limit 50 --output results/baseline_eval_results.json

# ── Step 2: Full Ablation Study (Table 3 in paper) ────────────────────────────
echo ""
echo "[Step 2] Running 6-Configuration Ablation Study (Table 3)..."
python scripts/run_ablation.py --limit 50 --output results/ablation_study_summary.json

# ── Step 3: Real Latency Benchmarking (Table 4 in paper) ──────────────────────
echo ""
echo "[Step 3] Measuring real component latencies (Mean, Std, P50, P95)..."
python scripts/measure_latency.py --runs 25 --output results/latency_benchmark.json

# ── Step 4: Categorical Error Analysis (Section 5 in paper) ───────────────────
echo ""
echo "[Step 4] Running Error Analysis on failure cases..."
python scripts/run_error_analysis.py --limit 50

# ── Step 5: Correction Quality Evaluation (Section 5B) ────────────────────────
echo ""
echo "[Step 5] Evaluating Correction Quality (1-5 rubric)..."
python scripts/evaluate_correction_quality.py

echo ""
echo "============================================================"
echo " Done! All publication-grade artifacts generated in results/:"
echo "   - results/baseline_eval_results.json"
echo "   - results/ablation_study_summary.json"
echo "   - results/latency_benchmark.json"
echo "   - results/error_analysis.json & results/error_analysis.html"
echo "   - results/correction_quality_report.json"
echo "============================================================"
