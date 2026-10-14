# TrustRAG — Related Work Section Draft

> Use this as a structural scaffold for Section 2 of the paper.
> Fill in the italic *[notes]* and add your own analysis.

---

## 2. Related Work

### 2.1 Fact Verification and Claim Checking

The FEVER shared task (Thorne et al., 2018) established fact verification as a core NLP problem,
requiring systems to classify claims as SUPPORTED, REFUTED, or NOT ENOUGH INFO against Wikipedia.
TrustRAG extends this three-class paradigm to the RAG setting, where the evidence corpus is a
private document collection rather than a general knowledge base.

FacTool (Chern et al., 2023) addresses factual error detection in LLM outputs using external
API-augmented verification. Unlike FacTool, TrustRAG is entirely self-contained — requiring no
external search APIs — and targets private document RAG rather than open-domain generation.

### 2.2 RAG Hallucination Evaluation

RAGTruth (Zhang et al., ACL 2024) introduced a large-scale hallucination corpus with
~18,000 naturally generated RAG responses and word-level human annotations across six LLMs.
We evaluate TrustRAG on the RAGTruth benchmark to provide the first direct comparison of
our pipeline against this naturalistic hallucination distribution.

HaluEval (Li et al., EMNLP 2023) provides 35,000 hallucination samples across QA, dialogue,
and summarization, generated via a sampling-then-filtering framework. We use the QA split as
a complementary benchmark to assess generalization across task types.

RAGAS (Es et al., 2023) proposes reference-free RAG evaluation metrics including faithfulness,
which we use as a baseline. Unlike RAGAS, TrustRAG performs claim-level verification and
produces corrected outputs rather than only a score.

### 2.3 Hallucination Detection and Mitigation

SelfCheckGPT (Manakul et al., EMNLP 2023) detects hallucinations by measuring consistency
across multiple stochastic samples from the same model. It requires no external knowledge
but cannot distinguish retrieval failure from factual error. TrustRAG uses retrieved evidence,
making it complementary rather than competing.

Self-RAG (Asai et al., NeurIPS 2023) trains models to generate reflection tokens that evaluate
retrieval quality and output faithfulness at inference time. Self-RAG requires model fine-tuning
and is not applicable to black-box LLMs. TrustRAG is model-agnostic and works as a post-hoc
layer over any generation pipeline.

Corrective RAG (Yan et al., ICLR 2024) introduces a retrieval evaluator that triggers web search
when retrieved documents are deemed irrelevant. Our calibrated abstention verifier (Section 3.4)
addresses a similar problem at the claim level: distinguishing retrieval failure from factual
contradiction.

AlignScore (Zha et al., ACL 2023) provides a unified factual consistency metric trained on 4.7M
examples across seven NLP tasks. MiniCheck (Tang et al., EMNLP 2024) refines this for
efficient sentence-level fact verification against grounding documents. We integrate MiniCheck
as our primary NLI backbone, replacing the generic cross-encoder used in our baseline.

### 2.4 Summary of Differences

| System | Requires fine-tuning | Model-agnostic | Handles private docs | Corrects output | Abstains on uncertain |
|---|---|---|---|---|---|
| FEVER | ✓ | ✗ | ✗ | ✗ | ✗ |
| SelfCheckGPT | ✗ | ✓ | ✗ | ✗ | ✗ |
| Self-RAG | ✓ | ✗ | ✓ | Partial | ✗ |
| CRAG | ✗ | ✓ | ✓ | ✗ | Partial |
| FacTool | ✗ | ✓ | ✗ | ✗ | ✗ |
| RAGAS | ✗ | ✓ | ✓ | ✗ | ✗ |
| **TrustRAG (Ours)** | **✗** | **✓** | **✓** | **✓** | **✓** |

---

## References to Add

- Thorne et al. (2018). FEVER: a large-scale dataset for fact extraction and VERification. NAACL.
- Chern et al. (2023). FacTool: Factuality Detection in Generative AI. arXiv:2307.13528.
- Zhang et al. (2024). RAGTruth: A Hallucination Corpus for Developing Trustworthy RAG. ACL.
- Li et al. (2023). HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs. EMNLP.
- Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.
- Manakul et al. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection. EMNLP.
- Asai et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique. NeurIPS.
- Yan et al. (2024). Corrective Retrieval Augmented Generation. ICLR.
- Zha et al. (2023). AlignScore: Evaluating Factual Consistency with a Unified Alignment Function. ACL.
- Tang et al. (2024). MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents. EMNLP.
