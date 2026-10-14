"""
TrustRAG Publication Figures Generator
======================================
Generates high-resolution (300 DPI), professionally styled, non-overlapping
vector figures for the IEEE Research Paper and Comprehensive Academic Report:
  1. fig1_pipeline_architecture.png: 3-Tier End-to-End System Architecture
  2. fig2_baseline_comparison.png: Head-to-Head Macro-F1 with 95% Bootstrap CIs
  3. fig3_ablation_study.png: 6-Config Ablation Progression & Abstention Rate
  4. fig4_error_taxonomy.png: Empirical Failure Mode Taxonomy Breakdown
  5. fig5_latency_profiling.png: Real Hardware Latency Profiling (P50 vs P95)
  6. fig6_rag_vs_trustrag.png: Standard RAG vs TrustRAG Closed-Loop Verification
"""

import os
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("results/figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Figure 1: 3-Tier System Architecture Diagram (Zero Text Collision)
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig1_architecture(filepath="results/figures/fig1_pipeline_architecture.png"):
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    ax.set_xlim(0, 11.0)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    # Tier Backgrounds
    # Stage 1: Ingestion & Indexing (y: 4.5 to 6.4)
    t1_bg = mpatches.FancyBboxPatch((0.15, 4.55), 10.7, 1.9, boxstyle="round,pad=0.06",
                                    facecolor="#f8fafc", edgecolor="#94a3b8", lw=1.2, linestyle="--")
    # Stage 2: Hybrid Retrieval & Context (y: 2.35 to 4.4)
    t2_bg = mpatches.FancyBboxPatch((0.15, 2.35), 10.7, 2.05, boxstyle="round,pad=0.06",
                                    facecolor="#ffffff", edgecolor="#94a3b8", lw=1.2, linestyle="--")
    # Stage 3: Generation, Verification & Repair (y: 0.15 to 2.2)
    t3_bg = mpatches.FancyBboxPatch((0.15, 0.15), 10.7, 2.05, boxstyle="round,pad=0.06",
                                    facecolor="#f8fafc", edgecolor="#94a3b8", lw=1.2, linestyle="--")
    ax.add_patch(t1_bg); ax.add_patch(t2_bg); ax.add_patch(t3_bg)

    ax.text(0.35, 6.28, "STAGE I: KNOWLEDGE INGESTION & HYBRID INDEXING", fontsize=8.5, fontweight="bold", color="#1e293b")
    ax.text(0.35, 4.23, "STAGE II: DUAL HYBRID RETRIEVAL (BM25 + FAISS + RRF) & NEURAL RERANKING", fontsize=8.5, fontweight="bold", color="#1e293b")
    ax.text(0.35, 2.03, "STAGE III: LLM SYNTHESIS, MULTI-ENGINE CLAIM VERIFICATION & REPAIR", fontsize=8.5, fontweight="bold", color="#1e293b")

    def draw_box(x, y, w, h, t1, t2="", col="#1e3a8a", title_col="white", sub_col="#e2e8f0", fs=8):
        box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", facecolor=col, edgecolor="white", lw=1.2)
        ax.add_patch(box)
        if t2:
            ax.text(x + w/2, y + h*0.64, t1, ha="center", va="center", fontsize=fs, fontweight="bold", color=title_col)
            ax.text(x + w/2, y + h*0.28, t2, ha="center", va="center", fontsize=fs-1.8, color=sub_col, style="italic")
        else:
            ax.text(x + w/2, y + h/2, t1, ha="center", va="center", fontsize=fs, fontweight="bold", color=title_col)

    # --- TIER 1 BOXES ---
    draw_box(0.35, 4.75, 1.85, 1.35, "Document Ingestion\n(PDF / DOCX)", "PyMuPDF Parser\nText Normalization", "#1e3a8a")
    draw_box(2.45, 4.75, 2.05, 1.35, "Sliding-Window\nChunking", "512 Chars Window\n64 Chars Overlap", "#1e3a8a")
    draw_box(4.75, 5.48, 2.65, 0.72, "FAISS Dense Index", "FlatIP | MiniLM 384-d Cosine", "#0284c7")
    draw_box(4.75, 4.65, 2.65, 0.72, "BM25Okapi Sparse Index", "Inverted Term Frequencies", "#0f766e")
    draw_box(7.65, 4.75, 2.95, 1.35, "Dual Hybrid Knowledge Store", "Dense Vectors + Sparse Inverted Index\nDeterministic Chunk Provenance", "#334155")

    # Connect Tier 1
    ax.annotate("", xy=(2.4, 5.42), xytext=(2.2, 5.42), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))
    ax.annotate("", xy=(4.7, 5.84), xytext=(4.5, 5.5), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))
    ax.annotate("", xy=(4.7, 5.01), xytext=(4.5, 5.34), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))
    ax.annotate("", xy=(7.6, 5.84), xytext=(7.4, 5.84), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))
    ax.annotate("", xy=(7.6, 5.01), xytext=(7.4, 5.01), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))

    # --- TIER 2 BOXES ---
    draw_box(0.35, 2.6, 1.85, 1.45, "User Query\nInput", "Natural Language\nInformation Query", "#0f172a")
    draw_box(2.45, 3.38, 2.05, 0.72, "FAISS Dense Search", "Cosine Sim Top-60", "#0284c7")
    draw_box(2.45, 2.55, 2.05, 0.72, "BM25 Sparse Search", "Term Match Top-60", "#0f766e")
    draw_box(4.75, 2.6, 2.1, 1.45, "Reciprocal Rank\nFusion (RRF)", "RRF(d)=sum(1/(60+r))\nDense + Sparse Merge", "#0369a1")
    draw_box(7.1, 2.6, 1.95, 1.45, "Cross-Encoder\nNeural Reranker", "ms-marco-MiniLM-L-6\nSelect Top-5 Chunks", "#1e40af")
    draw_box(9.3, 2.6, 1.35, 1.45, "Context\nBuilder", "IDs (E1..E5)\nProvenance", "#0f766e")

    # Connect Tier 2
    ax.annotate("", xy=(2.4, 3.74), xytext=(2.2, 3.4), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))
    ax.annotate("", xy=(2.4, 2.91), xytext=(2.2, 3.25), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))
    ax.annotate("", xy=(4.7, 3.32), xytext=(4.5, 3.5), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))
    ax.annotate("", xy=(4.7, 3.32), xytext=(4.5, 3.1), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))
    ax.annotate("", xy=(7.05, 3.32), xytext=(6.85, 3.32), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))
    ax.annotate("", xy=(9.25, 3.32), xytext=(9.05, 3.32), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))

    # Knowledge store to search arrows (vertical dotted)
    ax.annotate("", xy=(3.45, 4.15), xytext=(3.45, 4.75), arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=1.4, ls=":", mutation_scale=9))

    # --- TIER 3 BOXES ---
    draw_box(0.35, 0.35, 1.65, 1.5, "LLM Synthesis\n(Gemini 3.8)", "Temp=0.0 Grounded\nStrict Context Prompts", "#047857")
    draw_box(2.25, 0.35, 1.65, 1.5, "Atomic Claim\nDecomposition", "Sentence Boundary\nClause Segmentation", "#b45309")

    # Multi-engine verifier group box
    ax.add_patch(mpatches.FancyBboxPatch((4.15, 0.28), 4.35, 1.65, boxstyle="round,pad=0.05",
                                        facecolor="#fee2e2", edgecolor="#ef4444", lw=1.2))
    ax.text(6.32, 1.76, "MULTI-ENGINE VERIFICATION ENGINE", ha="center", fontsize=7.5, fontweight="bold", color="#991b1b")
    draw_box(4.3, 0.45, 1.25, 1.15, "MiniCheck\nCross-Enc", "DeBERTa-v3-L\nEntailment", "#b91c1c", fs=7)
    draw_box(5.7, 0.45, 1.35, 1.15, "Numeric Rule\nChecker", "0.08ms Analytical\nPercentages & Units", "#c2410c", fs=7)
    draw_box(7.2, 0.45, 1.15, 1.15, "Calibrated\nAbstention", "tau_rel=0.3\nUNVERIFIABLE", "#7c2d12", fs=7)

    draw_box(8.75, 0.35, 1.9, 1.5, "Evidence Repair &\nAudit Reporting", "Hedged Qualification\nProvenance Trace", "#4338ca")

    # Connect Tier 3
    ax.annotate("", xy=(2.2, 1.1), xytext=(2.0, 1.1), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))
    ax.annotate("", xy=(4.1, 1.1), xytext=(3.9, 1.1), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))
    ax.annotate("", xy=(8.7, 1.1), xytext=(8.5, 1.1), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=9))

    # Tier 2 Context Builder down to Tier 3 LLM synthesis
    ax.annotate("", xy=(0.8, 1.85), xytext=(9.9, 2.55),
                arrowprops=dict(arrowstyle="-|>", color="#047857", lw=1.4, ls="--",
                                connectionstyle="angle,angleA=0,angleB=90,rad=10", mutation_scale=9))

    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 1: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Figure 2: Head-to-Head Comparison Bar Chart with 95% CIs
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig2_baselines(filepath="results/figures/fig2_baseline_comparison.png"):
    fig, ax = plt.subplots(figsize=(6.8, 2.7), dpi=300)
    systems = [
        "Plain NLI\n(DeBERTa-small)",
        "MiniCheck NLI\n(Standalone)",
        "TrustRAG\n(DeBERTa-small)",
        "TrustRAG + MiniCheck\n(Full System)"
    ]
    f1_scores = [0.1111, 0.2500, 0.1111, 0.3000]
    ci_lows   = [0.0000, 0.1111, 0.0000, 0.0000]
    ci_highs  = [0.2500, 0.3333, 0.2500, 0.5556]

    err_low  = [f1 - lo for f1, lo in zip(f1_scores, ci_lows)]
    err_high = [hi - f1 for f1, hi in zip(f1_scores, ci_highs)]

    colors_list = ["#94a3b8", "#38bdf8", "#64748b", "#0284c7"]
    bars = ax.bar(systems, f1_scores, yerr=[err_low, err_high], capsize=5,
                  color=colors_list, edgecolor="#1e293b", linewidth=1.0, width=0.52)

    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.032, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Macro-F1 Score", fontsize=8.5, fontweight="bold")
    ax.set_title("Head-to-Head Macro-F1 with 95% Bootstrap Confidence Intervals (RAGTruth)",
                 fontsize=9.5, fontweight="bold", color="#0f172a", pad=8)
    ax.set_ylim(0, 0.70)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 2: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Figure 3: Ablation Study Dual-Axis Progression
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig3_ablation(filepath="results/figures/fig3_ablation_study.png"):
    fig, ax1 = plt.subplots(figsize=(6.8, 2.5), dpi=300)
    configs = [
        "Dense\n(FAISS)",
        "+ BM25\nHybrid",
        "+ CrossEnc\nReranker",
        "+ MiniCheck\nBackbone",
        "+ Abstention\n(Full TrustRAG)"
    ]
    f1_vals = [0.111, 0.111, 0.111, 0.250, 0.300]
    abstain_vals = [0.0, 0.0, 0.0, 0.0, 60.0]

    color1 = "#1e3a8a"
    color2 = "#d97706"

    ax1.plot(configs, f1_vals, marker="o", lw=2.2, color=color1, label="Macro-F1 Score")
    ax1.set_ylabel("Macro-F1", color=color1, fontsize=8.5, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.05, 0.36)

    ax2 = ax1.twinx()
    ax2.bar(configs, abstain_vals, color=color2, alpha=0.25, width=0.38, label="Abstention Rate (%)")
    ax2.set_ylabel("Abstention Rate (%)", color=color2, fontsize=8.5, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 100)

    for i, v in enumerate(f1_vals):
        ax1.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5, fontweight="bold", color=color1)

    ax1.set_title("Step-by-Step Architectural Ablation Impact on Macro-F1 & Abstention Rate",
                  fontsize=9.5, fontweight="bold", color="#0f172a", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 3: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Figure 4: Failure Mode Taxonomy Breakdown
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig4_error_taxonomy(filepath="results/figures/fig4_error_taxonomy.png"):
    fig, ax = plt.subplots(figsize=(6.5, 2.2), dpi=300)
    categories = ["Numerical\nContradiction", "Retrieval\nFailure", "Negation\nMissed", "Paraphrase\nError"]
    counts = [28, 9, 0, 0]
    pcts = [75.7, 24.3, 0.0, 0.0]
    bar_colors = ["#dc2626", "#ea580c", "#64748b", "#94a3b8"]

    bars = ax.barh(categories[::-1], counts[::-1], color=bar_colors[::-1], height=0.52)

    for bar, pct, cnt in zip(bars, pcts[::-1], counts[::-1]):
        if cnt > 0:
            ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height()/2,
                    f"{cnt} ({pct:.1f}%)", va="center", fontsize=8, fontweight="bold")
        else:
            ax.text(0.4, bar.get_y() + bar.get_height()/2, "0 (0.0%)", va="center", fontsize=7.5, color="#64748b")

    ax.set_xlabel("Count of Failure Cases", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, 34)
    ax.set_title("Failure Mode Taxonomy Breakdown (n=50 Sampled Verification Failures)",
                 fontsize=9.5, fontweight="bold", color="#0f172a", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 4: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Figure 5: Real Latency Profiling
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig5_latency(filepath="results/figures/fig5_latency_profiling.png"):
    fig, ax = plt.subplots(figsize=(6.8, 2.4), dpi=300)
    components = [
        "Numeric Contradiction Checker",
        "Plain NLI (DeBERTa-small)",
        "MiniCheck NLI (DeBERTa-large)",
        "Calibrated Abstention Verifier"
    ]
    p50 = [0.08, 107.66, 1373.65, 1505.83]
    p95 = [0.11, 112.30, 1630.38, 1840.19]

    y = np.arange(len(components))
    h = 0.34

    ax.barh(y - h/2, p50, height=h, color="#0284c7", label="P50 Latency (ms)")
    ax.barh(y + h/2, p95, height=h, color="#0f172a", label="P95 Latency (ms)")

    ax.set_yticks(y)
    ax.set_yticklabels(components, fontsize=8)
    ax.set_xlabel("Latency in Milliseconds (Log Scale)", fontsize=8.5, fontweight="bold")
    ax.set_xscale("log")
    ax.set_title("Real Hardware Latency Profiling on Consumer CPU (P50 vs P95)",
                 fontsize=9.5, fontweight="bold", color="#0f172a", pad=8)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 5: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Figure 6: Standard RAG vs TrustRAG Closed-Loop Verification
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig6_rag_vs_trustrag(filepath="results/figures/fig6_rag_vs_trustrag.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=300)
    for ax in axes:
        ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 5.2)
        ax.axis("off")

    # Standard RAG (Left)
    ax1 = axes[0]
    ax1.add_patch(mpatches.FancyBboxPatch((0.15, 0.1), 4.2, 4.9, boxstyle="round,pad=0.06",
                                         facecolor="#fef2f2", edgecolor="#ef4444", lw=1.4, ls="--"))
    ax1.text(2.25, 4.65, "Standard RAG Architecture", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#991b1b")
    ax1.text(2.25, 4.32, "(Open-Loop / Unverified)", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#b91c1c")
    
    stages_rag = [
        (3.4, "User Query", "#1e293b"),
        (2.4, "Vector Search (FAISS)", "#0284c7"),
        (1.4, "LLM Synthesis", "#047857"),
        (0.4, "Final Answer (Unverified Hallucination)", "#b91c1c")
    ]
    for y, text, col in stages_rag:
        ax1.add_patch(mpatches.FancyBboxPatch((0.55, y), 3.4, 0.62, boxstyle="round,pad=0.04", facecolor=col, edgecolor="white", lw=1.0))
        ax1.text(2.25, y+0.31, text, ha="center", va="center", fontsize=8.0, fontweight="bold", color="white")
    for y in [3.4, 2.4, 1.4]:
        ax1.annotate("", xy=(2.25, y-0.10), xytext=(2.25, y), arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.4))

    # TrustRAG (Right)
    ax2 = axes[1]
    ax2.add_patch(mpatches.FancyBboxPatch((0.15, 0.1), 4.2, 4.9, boxstyle="round,pad=0.06",
                                         facecolor="#f0fdf4", edgecolor="#22c55e", lw=1.4, ls="--"))
    ax2.text(2.25, 4.65, "TrustRAG Architecture", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#166534")
    ax2.text(2.25, 4.32, "(Closed-Loop / Evidence-Grounded)", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#15803d")

    stages_trust = [
        (3.5, "User Query -> Hybrid BM25+FAISS", "#0369a1"),
        (2.7, "LLM Grounded Synthesis", "#047857"),
        (1.9, "Atomic Claim Extraction", "#b45309"),
        (1.1, "Multi-Engine NLI + Numeric + Abstain", "#b91c1c"),
        (0.3, "Grounded Correction & Audit Trace", "#4338ca")
    ]
    for y, text, col in stages_trust:
        ax2.add_patch(mpatches.FancyBboxPatch((0.5, y), 3.5, 0.54, boxstyle="round,pad=0.04", facecolor=col, edgecolor="white", lw=1.0))
        ax2.text(2.25, y+0.27, text, ha="center", va="center", fontsize=7.8, fontweight="bold", color="white")
    for y in [3.5, 2.7, 1.9, 1.1]:
        ax2.annotate("", xy=(2.25, y-0.08), xytext=(2.25, y), arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.4))

    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 6: {filepath}")


if __name__ == "__main__":
    generate_fig1_architecture()
    generate_fig2_baselines()
    generate_fig3_ablation()
    generate_fig4_error_taxonomy()
    generate_fig5_latency()
    generate_fig6_rag_vs_trustrag()
    print("All 6 publication figures generated successfully with 300 DPI!")
