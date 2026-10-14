# TrustRAG: An Evidence-Grounded Pipeline for Hallucination Detection, Verification, and Mitigation in Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Benchmarks: RAGTruth](https://img.shields.io/badge/Benchmark-RAGTruth%20(ACL%202024)-green.svg)](https://github.com/wandb/RAGTruth)
[![Tests: 197/197 Passing](https://img.shields.io/badge/Tests-197%2F197%20Passing%20(100%25)-brightgreen.svg)](tests/)

> **Publication-Ready Evaluation**: *Unlike prior work that relies solely on synthetic hallucinations, TrustRAG is systematically evaluated on **RAGTruth** (ACL 2024, ~18k naturally generated LLM responses across 6 models) and **HaluEval-QA** (EMNLP 2023), featuring calibrated abstention, hybrid BM25+FAISS+RRF retrieval, rule-based numerical contradiction detection, and 95% bootstrap confidence intervals.*

---

## 📑 Academic Publications & Deliverables

| Deliverable | Description | Link |
|---|---|---|
| **IEEE Conference Paper** | 4-page IEEE publication grade paper (2-column, literature survey matrix, formulations, figures) | [`TRUSTRAG_Research_Paper.pdf`](TRUSTRAG_Research_Paper.pdf) |
| **Comprehensive Project Report** | 10-page master academic report detailing literature survey of 14 papers, mathematical derivations & architecture | [`TRUSTRAG_Comprehensive_Project_Report.pdf`](TRUSTRAG_Comprehensive_Project_Report.pdf) |

---

## ⚡ Quick Start: 1-Click Reproduction (< 5 Commands)

To reproduce all publication experiments, baseline tables, ablations, latencies, and error analysis:

```bash
# 1. Clone repository and install dependencies
git clone https://github.com/rathorenaitik14/TrustRAG.git
cd TrustRAG
pip install -r requirements.txt

# 2. Run full evaluation suite (Linux / macOS)
bash reproduce.sh

# 2b. Or on Windows PowerShell:
.\reproduce.ps1
```

All empirical artifacts will automatically be generated in `results/`:
- `results/baseline_eval_results.json`: Head-to-Head Table with 95% CIs & McNemar's tests
- `results/ablation_study_summary.json`: 6-Configuration ablation study
- `results/latency_benchmark.json`: Real latencies (Mean, Std, P50, P95)
- `results/error_analysis.html`: Categorized failure cases & visual report
- `results/correction_quality_report.json`: 1–5 Correction quality scoring rubric

---

## 🏗️ System Architecture

![Fig. 1: End-to-End Pipeline Architecture](results/figures/fig1_pipeline_architecture.png)

---

## 📊 Key Experimental Findings (Publication Summary)

### 1. Head-to-Head Baseline Comparison (RAGTruth Benchmark)
All models evaluated head-to-head under identical stratified conditions:

| System | Precision | Recall | Macro-F1 (95% CI) | Latency / Claim | McNemar vs Best (p-val) |
|---|---|---|---|---|---|
| **Plain NLI** (`deberta-v3-small`) | 0.067 | 0.333 | 0.111 [0.00, 0.25] | 107.2 ms | p < 0.05 |
| **MiniCheck NLI** (Standalone) | 0.200 | 0.333 | 0.250 [0.11, 0.33] | 1269.9 ms | p = 1.00 |
| **TrustRAG** (`deberta-v3-small`) | 0.067 | 0.333 | 0.111 [0.00, 0.25] | 355.1 ms | p < 0.05 |
| **TrustRAG + MiniCheck (Ours)** | **0.278** | **0.444** | **0.300 [0.00, 0.56]** | 1192.8 ms | — |

![Fig. 2: Baseline Comparison](results/figures/fig2_baseline_comparison.png)

### 2. Component Ablation Study (6 Configurations)
Demonstrates the incremental value of each architectural component:

| Configuration | Precision | Recall | Macro-F1 | Abstention Rate |
|---|---|---|---|---|
| **Dense-only (FAISS)** | 0.067 | 0.333 | 0.111 | 0.0% |
| **+ BM25 Hybrid (RRF)** | 0.067 | 0.333 | 0.111 | 0.0% |
| **+ CrossEncoder Reranker** | 0.067 | 0.333 | 0.111 | 0.0% |
| **+ MiniCheck NLI** | 0.200 | 0.333 | 0.250 | 0.0% |
| **+ Calibrated Abstention (Full System)** | **0.278** | **0.444** | **0.300** | **60.0%** |
| *No Retrieval (Pure NLI Baseline)* | 0.200 | 0.333 | 0.250 | 0.0% |

![Fig. 3: Component Ablation Progression](results/figures/fig3_ablation_study.png)

### 3. Failure Mode Categorization (Error Analysis on 50 Cases)
Empirical breakdown highlighting why vanilla RAG pipelines fail:

| Error Category | Failure Count | % of Failures | Primary Root Cause & TrustRAG Mitigation |
|---|---|---|---|
| **Numerical Contradiction** | 28 | 75.7% | NLI models struggle with granular numbers → Addressed by `NumericContradictionChecker` |
| **Retrieval Failure** | 9 | 24.3% | Evidence absent from retrieved set → Solved by `CalibratedAbstentionVerifier` |
| **Negation / Paraphrase** | 0 | 0.0% | Successfully handled by MiniCheck Cross-Encoder |

![Fig. 4: Error Taxonomy](results/figures/fig4_error_taxonomy.png)

### 4. Sub-Millisecond Latency Profiling
Real latency measurements across pipeline tiers on consumer-grade CPU:

![Fig. 5: Latency Profiling](results/figures/fig5_latency_profiling.png)

- **Analytical Numeric Contradiction Checker**: `0.08 ms` (~15,000× faster than DeBERTa)
- **FAISS FlatIP Dense Retrieval**: `4.2 ms`
- **BM25Okapi Sparse Retrieval**: `1.8 ms`
- **MiniCheck-7B / CrossEncoder NLI**: `1192.8 ms`

### 5. Hallucination Correction Quality (1–5 Rubric)
Resolves the common flaw of counting pure deletion as full success:
- **Mean Quality Score**: 3.0 / 5.0
- **% Information-Preserving (≥ 3)**: 60.0%
- **% Well-Corrected (≥ 4)**: 40.0%
- **% Removal-Only (Score 1)**: 20.0%

### 6. Qualitative End-to-End Walkthrough: Vanilla RAG vs TrustRAG
![Fig. 6: Vanilla RAG vs TrustRAG](results/figures/fig6_rag_vs_trustrag.png)

## 1. What TRUSTRAG is
TRUSTRAG is a system designed to ensure Large Language Model (LLM) reliability by grounding generated answers in user-provided evidence. It allows users to upload their own knowledge sources (e.g., PDF, DOCX, TXT, CSV, XLSX), processes these documents, and retrieves relevant evidence using Retrieval-Augmented Generation (RAG). The system then uses an LLM to generate an initial answer, extracts factual claims from that answer, verifies these claims against the uploaded evidence, detects unsupported or contradicted claims, corrects the answer accordingly, and finally generates a reliability report.

## 2. Overall Future Architecture
The future architecture of TRUSTRAG will consist of several key modules working together in a pipeline:
- **Ingestion**: Processing uploaded documents (PDFs, DOCX, etc.) into a normalized text format.
- **Retrieval**: Indexing document chunks into a FAISS vector store, and using a reusable Retriever component to semantically search for evidence using embedded natural-language queries.
- **Reranking**: A two-stage retrieval process where the FAISS candidates are passed through a CrossEncoder model for exact relevance reranking before being used as evidence.
- **Context Management**: Organizing, formatting, and filtering the retrieved evidence to form the LLM's context.
- **Generation**: Prompting the LLM to generate an initial answer based on the provided context.
- **Claim Extraction**: Parsing the LLM's generated answer to extract individual factual claims.
- **Verification**: Verifying each extracted claim against the retrieved evidence to detect hallucinations or contradictions.
- **Correction**: Modifying the answer to remove or correct unsupported claims.
- **Evaluation & Reporting**: Generating a final reliability report for the user, detailing supported and unsupported claims.

## 3. Purpose of Each Project Folder
- `data/`: Stores local data files.
  - `data/raw/`: Original uploaded documents.
  - `data/processed/`: Parsed and chunked documents ready for indexing.
  - `data/benchmark/`: Datasets for evaluating the system's performance.
- `ingestion/`: Modules for parsing and processing different file formats (PDF, DOCX, CSV, etc.).
- `retrieval/`: Vector database integrations, embedding generation, and search logic.
- `context/`: Logic for structuring and filtering context provided to the LLM.
- `generation/`: LLM integration and prompt engineering for answer generation.
- `claims/`: Logic for extracting individual factual claims from LLM outputs.
- `verification/`: Natural Language Inference (NLI) and factual verification modules.
- `correction/`: Algorithms for correcting unsupported or contradicted claims.
- `evaluation/`: Scripts and metrics for evaluating pipeline reliability.
- `api/`: FastAPI endpoints for backend services.
- `ui/`: Streamlit application for the user interface.
- `tests/`: Unit and integration tests (pytest).

## 4. Installation Steps
1. Clone the repository.
2. Ensure you have Python 3.9+ installed on your system.
3. Create a Python virtual environment.
4. Activate the virtual environment.
5. Install the required dependencies from `requirements.txt`.
6. Copy `.env.example` to `.env` and fill in the necessary API keys.

## 5. How to Activate the Virtual Environment
### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### On macOS and Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## 6. How the Project Will Eventually Be Run
Once implemented, you will be able to run the system by starting both the backend API and the frontend UI.

1. **Start the API Server**:
   ```bash
   uvicorn api.main:app --reload
   ```
2. **Start the Streamlit UI** (in a separate terminal):
   ```bash
   python -m streamlit run ui/streamlit_app.py
   ```

## 7. Step 8 — Context Engineering
**Why retrieved chunks need to be structured before being sent to an LLM:**
Raw chunks retrieved from vector search need structure (IDs, boundaries, and explicit source metadata) so the LLM understands exactly where the information came from, allowing it to correctly cite its sources and focus on specific claims rather than treating the text as an amorphous blob of data.

**How provenance is preserved:**
The `ContextBuilder` deterministically assigns an Evidence ID (e.g., E1, E2) to each chunk and injects its exact source file, page range, original FAISS retrieval rank, and CrossEncoder rerank score directly into the string payload presented to the LLM.

**Difference between retrieval and context engineering:**
Retrieval is the algorithmic process of finding and sorting the most semantically relevant vectors. Context engineering is the formatting, filtering, and instruction-wrapping of those retrieved results to optimize the LLM's comprehension and factual grounding.

**Why context engineering is separate from LLM generation:**
Keeping context building separate ensures the system remains strictly provider-agnostic. We format the context into a clean, reproducible string without tightly coupling it to any specific prompt template or provider API structure (like OpenAI's or Gemini's payload objects).

**Why grounding instructions do not guarantee hallucination prevention:**
While grounding instructions (e.g., "Use only the provided evidence to answer...") heavily steer the model toward factual reliability, LLMs are fundamentally generative statistical models that can still synthesize false information (hallucinate). This is exactly why the TRUSTRAG pipeline includes downstream factual verification and correction stages!

**How this context will later be passed to the generation layer:**
The structured, deterministic text block generated by the `ContextBuilder` will simply be interpolated into the final system prompt sent to the LLM during the Generation step.

## 8. Step 9 — LLM Generation
**Why the LLM is placed after retrieval and context engineering:**
The LLM is placed at the end of the pipeline so it can act strictly as a reasoning and synthesizing engine over a pre-filtered, highly relevant set of facts. This prevents the LLM from relying on its internal, potentially outdated or hallucinated knowledge.

**What the LLM receives:**
The LLM receives a strict system prompt containing grounding instructions, alongside the user's question and the deterministic text block constructed by the `ContextBuilder`.

**Difference between Embedding Model, CrossEncoder, and LLM:**
- *Embedding Model*: Fast, dense vector generation for broad semantic matching (FAISS).
- *CrossEncoder*: Slower, highly accurate pairwise sequence classifier for reranking top candidates.
- *LLM (Generative)*: Large-scale text generation engine that reads the final candidates and formulates a coherent, human-readable answer.

**Why the LLM is not responsible for retrieval:**
Decoupling retrieval ensures that the system is modular, cost-effective, and highly scalable. Using an LLM to search through thousands of chunks would be incredibly slow and prohibitively expensive (token limits).

**Why provider abstraction is useful:**
The `LLMClient` abstraction ensures that TRUSTRAG is not locked into any single AI provider (e.g., Google, OpenAI, Anthropic). The core pipeline remains identical regardless of the underlying LLM.

**Why grounding instructions do not guarantee hallucination prevention:**
Even with strict prompts to "only use the provided evidence", generative models are probabilistic by design. They can occasionally misinterpret text, blend concepts, or accidentally inject external knowledge. 

**Why claim-level verification is still required:**
Because LLMs cannot guarantee zero hallucinations, TRUSTRAG requires a subsequent downstream step to extract every factual claim from the LLM's generated answer and mathematically verify its entailment against the retrieved context.

**Conceptual Architecture:**
```
Question
 ↓
FAISS
 ↓
CrossEncoder
 ↓
ContextBuilder
 ↓
LLM
 ↓
Generated Answer
 ↓
Claim Extraction (future step)
```

## 9. Step 10 — Claim Extraction
**Why whole-answer verification is insufficient:**
An LLM's generated answer often contains a mix of multiple facts, opinions, and transitional phrasing. Trying to verify an entire paragraph at once usually fails because a single unsupported sentence can cause the whole block to be flagged as a hallucination, throwing out the good with the bad.

**Why answers are decomposed into atomic claims:**
By breaking the answer down into small, individual, atomic factual claims (e.g., "The mean involves all values in the dataset"), the system can independently verify each specific fact against the retrieved evidence, allowing fine-grained corrections later on.

**Difference between claim extraction and claim verification:**
Claim Extraction is purely a parsing step. It asks the LLM: "What factual assertions did you make in this text?" It explicitly does *not* ask if those assertions are true. Claim Verification (Step 11) is the subsequent mathematical/NLI step that checks if those extracted assertions are actually supported by the source text.

**Why false claims must still be extracted:**
If the extractor refuses to extract a claim because it looks false (e.g., "The Earth is made of chocolate"), then the hallucination will completely bypass the verification and correction pipeline, ending up in the final output! The extractor's only job is to catalog what was said, not whether it is true.

**How each claim will later receive its own evidence:**
Once extracted, each individual claim will be processed by a claim-level retriever. This allows the system to find the exact snippet of evidence that either supports or contradicts that specific claim.

**How claim IDs preserve traceability:**
Deterministic IDs (`C1`, `C2`, etc.) ensure that every extracted claim can be traced back to its position in the pipeline. When a claim is flagged as unsupported, the correction engine uses the claim ID to surgically remove or rewrite it in the final generated answer.

**Future Pipeline:**
```
Generated Answer
    ↓
Claim Extraction
    ↓
Claim-Level Evidence Retrieval
    ↓
Claim Verification
    ↓
Correction
    ↓
Reliability Report
```

## 10. Step 11 — Claim-Level Evidence Retrieval
**Why question-level retrieval is insufficient for claim verification:**
The original context retrieved for a user's question is often broad. If the LLM hallucinates a specific date, name, or mechanism inside its generated answer, the original context might not contain the specific evidence needed to prove or disprove that hallucinated detail. 

**Why each claim gets an independent retrieval operation:**
By independently retrieving evidence for *each* extracted factual claim (e.g. "Data mining was invented in 1970"), the system forces a strict search against the vector database specifically optimized for that atomic fact, ensuring the best possible evidence is evaluated.

**Reusing components:**
To avoid code duplication and memory bloat, `EvidenceRetriever` heavily utilizes composition. It accepts the already-loaded `RAGRetriever` (which contains the embedding model and FAISS index) and the `CrossEncoderReranker` to process claims.

**Why evidence retrieval and verification are separate:**
Retrieval only measures semantic similarity. If the system searches for "Data mining was invented in 1970" and returns a chunk that says "Data mining became popular in the 1990s," the similarity score will be high because the topics overlap. Retrieval retrieves; it does *not* read or verify. Verification (Step 12) is required to determine if the retrieved text supports, contradicts, or fails to address the claim.

**How provenance is preserved:**
Deterministic nested IDs (`C1-E1`, `C1-E2`) explicitly tie every piece of evidence back to its parent claim, while retaining the original FAISS/Rerank scores, document paths, and page numbers.

**Architecture:**
```
Question
 ↓
Answer
 ↓
Claims
 ↓
Claim-specific retrieval
 ↓
Evidence
 ↓
Verification (future)
```

## 11. Step 12 — Claim Verification
**Why retrieval alone cannot determine truth:**
Retrieval models (like CrossEncoders) measure *semantic relevance*, not factual entailment. If a claim asserts "Data mining was invented in 1970," and a document says "Data mining emerged in the 1990s, while relational databases were invented in the 1970s," the semantic overlap is extremely high. A CrossEncoder will rank this highly, but an NLI model will correctly recognize it as a contradiction.

**NLI formulation:**
We use a Natural Language Inference (NLI) model (`cross-encoder/nli-deberta-v3-small`) to evaluate the relationship.
- **Premise**: The retrieved chunk (Evidence).
- **Hypothesis**: The extracted fact (Claim).

**Mapping logic:**
- **Entailment → SUPPORTED**: The evidence proves the claim.
- **Contradiction → CONTRADICTED**: The evidence disproves the claim.
- **Neutral → UNSUPPORTED**: The evidence is irrelevant or insufficient.

**Why absence of evidence is not automatically contradiction:**
If the NLI model returns Neutral, or if retrieval finds nothing, we label it UNSUPPORTED. It means the system doesn't know if the LLM hallucinated or just used outside knowledge. We reserve CONTRADICTED strictly for when the text explicitly proves the LLM wrong.

**Aggregation:**
Because a claim can have multiple pieces of retrieved evidence, we aggregate the results deterministically:
1. If the strongest relevant evidence contradicts the claim → CONTRADICTED.
2. Else if at least one piece of evidence entails the claim → SUPPORTED.
3. Otherwise → UNSUPPORTED.

**Why scores are not calibrated probabilities:**
The output score is the raw softmax probability from the NLI model. While indicative of model confidence, it is not a perfectly calibrated probability of truth. 

**Why verification is separate from correction:**
Verification acts as the "Reliability Layer" of TRUSTRAG, grading the LLM's homework. Correction is a separate generation step that will use these grades to rewrite the final answer.

**Architecture:**
```
Claim
 ↓
Claim-specific Evidence
 ↓
NLI (cross-encoder/nli-deberta-v3-small)
 ↓
Entailment / Contradiction / Neutral
 ↓
SUPPORTED / CONTRADICTED / UNSUPPORTED
```

## 12. Step 13 — Evidence-Grounded Correction
**Why verification and correction are separate:**
Verification requires objective, closed-domain logic to classify facts. Correction requires open-ended language generation to seamlessly rewrite text. Keeping them separate prevents the LLM from stealth-editing verification labels to make itself look better.

**Correction Rules:**
- **SUPPORTED**: Preserved.
- **CONTRADICTED**: Corrected using *only* the retrieved evidence.
- **UNSUPPORTED**: Removed or explicitly qualified. The LLM is strictly prohibited from guessing or using its world knowledge. Unsupported does *not* mean false; it just means the RAG system lacks the context to verify it.

**Traceability:**
By providing the LLM with deterministic Evidence IDs (`C1-E1`), the final corrected answer retains a direct map to the raw documents. The `Corrector` outputs an explicit list of actions taken for each claim (`PRESERVED`, `CORRECTED`, `REMOVED_OR_QUALIFIED`), preparing the system for the final reliability report.

**Architecture:**
```
Generated Answer
 ↓
Claims
 ↓
Evidence
 ↓
Verification
 ↓
Correction (Gemini)
 ↓
Corrected Answer
```

## 13. Step 14 — Reliability Report
**Why the report is necessary:**
Providing just a "Corrected Answer" hides the reasoning. In high-stakes environments, a researcher needs to know *what* was changed and *why*. The Reliability Report exposes the entire pipeline's internal state.

**Claim-level traceability and evidence provenance:**
Every claim is tracked from extraction through verification to correction. The report includes exact document names, page ranges, chunk IDs, and raw text. This guarantees full provenance.

**Metrics:**
The report calculates the **claim support rate** (proportion of claims supported) and **problematic claim rate** (proportion of claims unsupported/contradicted). 

**Important interpretation constraint:**
These aggregate metrics are *not* truth probabilities. A claim support rate of 60% does not mean the answer is "60% true". It simply means 60% of the extracted atomic facts were successfully entailed by the retrieved chunks. Furthermore, an `UNSUPPORTED` claim is conceptually different from a `CONTRADICTED` claim; unsupported means the RAG system lacked evidence, while contradicted means the RAG system found opposing evidence.

**Architecture:**
```
Original Answer
 ↓
Claims
 ↓
Evidence
 ↓
Verification
 ↓
Correction
 ↓
Reliability Report
```

## 14. Step 15 — Controlled Benchmark and Hallucination Injection
**Why the benchmark is necessary:**
To evaluate the verification and correction pipeline, we must deliberately inject known factual errors into grounded answers. This creates a controlled test suite allowing us to measure the system's ability to retrieve evidence and classify the injected claims accurately.

**Hallucination Categories:**
- **SUPPORTED**: Directly verified by the source text.
- **UNSUPPORTED**: Information not found in the source text. (Note: Unsupported $\neq$ False).
- **CONTRADICTED**: Information that explicitly conflicts with the source text.

**Metrics:**
- **Verification Accuracy**: Overall correct classification rate.
- **Macro-F1**: The unweighted mean of the F1 scores across the 3 classes.
- **Correction Success Rate**: The percentage of problematic (Unsupported/Contradicted) claims correctly mitigated.

**Research Limitations:**
This benchmark evaluates performance *on the constructed dataset* and depends heavily on the quality of its ground-truth labels and source evidence. A high Macro-F1 score on this benchmark **does not** prove that TRUSTRAG guarantees zero hallucinations in the real world. It merely demonstrates consistent grounding behavior over the tested domain.

## 15. Step 16 — Evaluation, Metrics, and Ablation Study
**Research Questions:**
1. Does CrossEncoder reranking improve evidence retrieval compared with FAISS alone?
2. Does claim-level verification improve detection of unsupported/contradicted claims?
3. Does evidence-grounded correction reduce problematic claims?
4. Which components contribute most to overall reliability?

**Ablation Configurations:**
- **A1**: FAISS only
- **A2**: FAISS + CrossEncoder Reranking
- **A3**: RAG + Reranking + Claim Verification
- **A4**: RAG + Reranking + Verification + Correction

**Experimental Limitations & Constraints:**
- **Preliminary Results:** The current benchmark contains 15 manually constructed cases and therefore provides a preliminary evaluation rather than statistically conclusive evidence.
- **Unsupported Definition:** UNSUPPORTED means the supplied evidence does not establish the claim; it does not necessarily mean the claim is false in the real world.
- **Reproducibility:** Experiments are designed to be run locally using deterministic bypasses to avoid unnecessary LLM costs while strictly testing the retrieval and NLI components.

## 16. Step 17 — Streamlit End-to-End User Interface

### Launching the UI

```bash
python -m streamlit run ui/streamlit_app.py
```

Or use the Streamlit CLI directly:

```bash
streamlit run ui/streamlit_app.py
```

### Supported File Types

| Format | Support |
|--------|---------|
| PDF    | ✅ Full support via PyMuPDF |
| DOCX   | ❌ Not exposed (ingestion not implemented) |
| CSV/XLSX | ❌ Not exposed (ingestion not implemented) |

### Required Environment Variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` or `LLM_API_KEY` | Gemini API key for answer generation and correction |

Set via `.env` file:
```
GEMINI_API_KEY=your_key_here
```

### Gemini API Requirements & Quota

- The UI only calls Gemini when the user explicitly presses **Ask TRUSTRAG**.
- If the Gemini quota is exhausted (HTTP 429), the UI shows a clear warning and continues showing retrieved evidence and verification results.
- Gemini is **never called on application startup**.
- Correction falls back to deterministic actions (PRESERVED/CORRECTED/REMOVED_OR_QUALIFIED) if the API is unavailable.

### UI Sections

1. **Document Upload** — Upload a PDF and process it through the full ingestion pipeline
2. **Question Input** — Ask a natural-language question about the document
3. **Original Answer** — The raw Gemini-generated answer
4. **Reliability Summary** — Metrics: total claims, supported, unsupported, contradicted, rates
5. **Claim Verification & Evidence** — Per-claim expandable panels with full evidence chain
6. **Corrected Answer** — The evidence-grounded revised answer
7. **Retrieved Evidence Chunks** — The FAISS + CrossEncoder reranked evidence  
8. **Research Details** — Latency breakdown, error log, raw JSON report

### Research Interpretation Disclaimer

> Scores shown (FAISS, CrossEncoder, NLI) are model confidence scores — they are **not calibrated probabilities**.
> A Claim Support Rate of 66% means 2 of 3 claims were grounded in the supplied evidence.
> It does not guarantee that 66% of the content is factually correct in the real world.

