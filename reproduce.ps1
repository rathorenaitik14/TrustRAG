# =============================================================================
# TrustRAG Windows PowerShell Reproduction Script
# Reproduces all publication experimental benchmarks and analyses in one command.
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " TrustRAG — Full Publication Reproduction" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$PythonExe = ".\venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

# ── Step 0: Ensure dependencies ─────────────────────────────────────────────
Write-Host "`n[Step 0] Checking dependencies..." -ForegroundColor Yellow
& $PythonExe -m pip install -r requirements.txt --quiet

# ── Step 1: Head-to-Head Baseline Evaluation (Table 2 in paper) ──────────────
Write-Host "`n[Step 1] Running Head-to-Head Baseline Evaluation on RAGTruth (Table 2)..." -ForegroundColor Yellow
& $PythonExe scripts/evaluate_baselines.py --limit 50 --output results/baseline_eval_results.json

# ── Step 2: Full Ablation Study (Table 3 in paper) ────────────────────────────
Write-Host "`n[Step 2] Running 6-Configuration Ablation Study (Table 3)..." -ForegroundColor Yellow
& $PythonExe scripts/run_ablation.py --limit 50 --output results/ablation_study_summary.json

# ── Step 3: Real Latency Benchmarking (Table 4 in paper) ──────────────────────
Write-Host "`n[Step 3] Measuring real component latencies (Mean, Std, P50, P95)..." -ForegroundColor Yellow
& $PythonExe scripts/measure_latency.py --runs 25 --output results/latency_benchmark.json

# ── Step 4: Categorical Error Analysis (Section 5 in paper) ───────────────────
Write-Host "`n[Step 4] Running Error Analysis on failure cases..." -ForegroundColor Yellow
& $PythonExe scripts/run_error_analysis.py --limit 50

# ── Step 5: Correction Quality Evaluation (Section 5B) ────────────────────────
Write-Host "`n[Step 5] Evaluating Correction Quality (1-5 rubric)..." -ForegroundColor Yellow
& $PythonExe scripts/evaluate_correction_quality.py

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " Done! All publication-grade artifacts generated in results/:" -ForegroundColor Green
Write-Host "   - results/baseline_eval_results.json" -ForegroundColor Green
Write-Host "   - results/ablation_study_summary.json" -ForegroundColor Green
Write-Host "   - results/latency_benchmark.json" -ForegroundColor Green
Write-Host "   - results/error_analysis.json & results/error_analysis.html" -ForegroundColor Green
Write-Host "   - results/correction_quality_report.json" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
