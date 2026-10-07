"""
TRUSTRAG — Evidence-Grounded LLM Reliability System
Streamlit End-to-End User Interface (Step 17)

Launch with:
    python -m streamlit run ui/streamlit_app.py
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add project root to sys.path so all TRUSTRAG modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from ui.session_state import init_state, reset_query_state, reset_document_state
from ui.components import (
    is_supported_file,
    get_status_icon,
    get_correction_label,
    format_score_note,
    summarise_metrics,
    build_evidence_chain,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TRUSTRAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ TRUSTRAG")
    st.caption("Evidence-Grounded LLM Reliability System")
    st.divider()

    st.subheader("System Models")
    st.markdown("""
| Component | Model |
|-----------|-------|
| Embedder | `all-MiniLM-L6-v2` |
| Reranker | `ms-marco-MiniLM-L-6-v2` |
| Verifier | `nli-deberta-v3-small` |
| Generator | `Gemini` |
""")

    st.divider()
    st.subheader("Pipeline")
    st.markdown("""
Ingestion → Chunking → Embeddings → Retrieval → Reranking → Generation → Claims → Verification → Correction
""")

    st.divider()
    st.subheader("Configuration")
    retrieval_k = st.slider("Top-K Retrieval (FAISS)", 5, 30, 10, key="retrieval_k_slider")
    rerank_k = st.slider("Top-K Reranking", 3, 15, 5, key="rerank_k_slider")
    st.session_state["retrieval_k"] = retrieval_k
    st.session_state["rerank_k"] = rerank_k

    st.divider()
    st.caption("**Supported file types:** PDF")
    st.caption("**Note:** Scores shown are model confidence scores — not calibrated probabilities.")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔍 TRUSTRAG")
st.subheader("Evidence-Grounded LLM Reliability System")
st.markdown(
    "Ask questions about your documents and inspect the evidence, claims, "
    "verification decisions, and corrected answer."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Document Upload
# ─────────────────────────────────────────────────────────────────────────────
st.header("📄 1. Upload Document")
st.caption("**Supported formats:** PDF only")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    help="Only PDF files are supported. Other file types are not processed.",
)

if uploaded_file is not None:
    # Detect new document upload
    if st.session_state["uploaded_file_name"] != uploaded_file.name:
        reset_document_state()
        st.session_state["uploaded_file_name"] = uploaded_file.name

        if not is_supported_file(uploaded_file.name):
            st.error(f"❌ Unsupported file type: `{uploaded_file.name}`. Only PDF files are accepted.")
        else:
            with st.status("Processing document…", expanded=True) as doc_status:
                try:
                    # ── Save to temp file ──────────────────────────────────
                    st.write("💾 Saving uploaded file…")
                    raw_dir = Path("data/raw")
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    save_path = raw_dir / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # ── Ingestion ──────────────────────────────────────────
                    st.write("📖 Extracting text from PDF…")
                    from ingestion.pdf_loader import load_pdf
                    pages = load_pdf(str(save_path))
                    if not pages:
                        st.error("❌ No text could be extracted from this PDF.")
                        st.stop()

                    # ── Chunking ───────────────────────────────────────────
                    st.write("✂️  Chunking document…")
                    from ingestion.chunker import chunk_pages
                    chunks = chunk_pages(pages)
                    if not chunks:
                        st.error("❌ No chunks produced from this document.")
                        st.stop()

                    # ── Embeddings + FAISS ─────────────────────────────────
                    st.write("🔢 Generating embeddings and building FAISS index…")
                    from retrieval.embeddings import embed_chunks
                    from retrieval.faiss_store import build_faiss_index, save_faiss_index
                    import json, numpy as np

                    processed_dir = Path("data/processed")
                    processed_dir.mkdir(parents=True, exist_ok=True)
                    stem = save_path.stem

                    # Save chunks as JSONL (flat dicts for disk storage)
                    chunks_jsonl_path = processed_dir / f"{stem}_chunks.jsonl"
                    with open(chunks_jsonl_path, "w", encoding="utf-8") as f:
                        for c in chunks:
                            f.write(json.dumps({
                                "chunk_id": c.chunk_id,
                                "document_id": c.document_id,
                                "document_name": c.document_name,
                                "text": c.text,
                                "page_start": c.page_start,
                                "page_end": c.page_end,
                                "source_path": c.source_path,
                            }) + "\n")

                    # embed_chunks expects ChunkRecord objects (with .text attribute)
                    embeddings, emb_metadata = embed_chunks(chunks)
                    dim = emb_metadata["dimension"]
                    index = build_faiss_index(embeddings)

                    faiss_path = processed_dir / f"{stem}_chunks_faiss.index"
                    save_faiss_index(index, str(faiss_path))

                    # Build index_mapping
                    index_mapping = {str(i): c.chunk_id for i, c in enumerate(chunks)}
                    meta = {"index_mapping": index_mapping, "dim": dim}
                    meta_path = processed_dir / f"{stem}_chunks_embeddings_metadata.json"
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f)

                    # ── Build retriever components ─────────────────────────
                    st.write("🗂️  Initialising retriever…")
                    from retrieval.retriever import RAGRetriever
                    from retrieval.reranker import CrossEncoderReranker
                    from claims.evidence_retriever import EvidenceRetriever
                    from verification.nli_verifier import NLIVerifier
                    from correction.gemini_corrector import GeminiCorrector
                    from evaluation.reliability_report import ReliabilityReportGenerator

                    retriever = RAGRetriever(
                        str(faiss_path), str(meta_path), str(chunks_jsonl_path)
                    )
                    reranker = CrossEncoderReranker()
                    ev_retriever = EvidenceRetriever(
                        retriever, reranker,
                        default_retrieval_k=retrieval_k,
                        default_evidence_top_k=rerank_k
                    )
                    verifier = NLIVerifier(model_name="cross-encoder/nli-deberta-v3-small")

                    try:
                        corrector = GeminiCorrector()
                    except Exception:
                        corrector = None  # Gemini unavailable — handled later

                    report_gen = ReliabilityReportGenerator()

                    # ── Save to session state ──────────────────────────────
                    st.session_state["retriever"] = retriever
                    st.session_state["reranker"] = reranker
                    st.session_state["ev_retriever"] = ev_retriever
                    st.session_state["verifier"] = verifier
                    st.session_state["corrector"] = corrector
                    st.session_state["report_generator"] = report_gen
                    st.session_state["document_pages"] = len(pages)
                    st.session_state["document_chunks"] = len(chunks)
                    st.session_state["embedding_dim"] = dim

                    doc_status.update(label="✅ Document ready!", state="complete")

                except FileNotFoundError as e:
                    st.error(f"❌ File error: {e}")
                except ValueError as e:
                    st.error(f"❌ Validation error: {e}")
                except RuntimeError as e:
                    st.error(f"❌ Processing error: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error during document processing. Please try again.")

# ── Document info card ─────────────────────────────────────────────────────
if st.session_state.get("retriever") is not None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Document", st.session_state["uploaded_file_name"])
    col2.metric("Pages", st.session_state["document_pages"])
    col3.metric("Chunks", st.session_state["document_chunks"])
    col4.metric("Embedding Dim", st.session_state["embedding_dim"])

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Question
# ─────────────────────────────────────────────────────────────────────────────
st.header("❓ 2. Ask a Question")

question = st.text_area(
    "Ask a question about your document",
    placeholder="e.g. What is the goal of data mining?",
    height=80,
)

ask_btn = st.button("🚀 Ask TRUSTRAG", type="primary", disabled=(st.session_state.get("retriever") is None))

if ask_btn and question.strip():
    reset_query_state()
    st.session_state["question"] = question.strip()

    retriever = st.session_state["retriever"]
    reranker = st.session_state["reranker"]
    ev_retriever = st.session_state["ev_retriever"]
    verifier = st.session_state["verifier"]
    corrector = st.session_state["corrector"]
    report_gen = st.session_state["report_generator"]
    k = st.session_state["retrieval_k"]
    rk = st.session_state["rerank_k"]
    errors = {}
    latency = {}

    progress = st.progress(0, text="Starting pipeline…")

    # ── Stage 1: Retrieve ──────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        retrieved = retriever.retrieve(question.strip(), k=k)
        latency["FAISS Retrieval"] = time.perf_counter() - t0
        st.session_state["retrieved_chunks"] = retrieved
        progress.progress(12, text="✅ Evidence retrieved")
    except Exception as e:
        errors["retrieval"] = str(e)
        st.error(f"❌ Retrieval failed: {e}")
        st.stop()

    # ── Stage 2: Rerank ────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        reranked = reranker.rerank(question.strip(), retrieved, top_k=rk)
        latency["CrossEncoder Reranking"] = time.perf_counter() - t0
        st.session_state["reranked_chunks"] = reranked
        progress.progress(25, text="✅ Evidence reranked")
    except Exception as e:
        errors["reranking"] = str(e)
        reranked = retrieved[:rk]
        st.session_state["reranked_chunks"] = reranked
        st.warning(f"⚠️ Reranking unavailable: {e}. Using raw FAISS order.")

    # ── Stage 3: Context ───────────────────────────────────────────────────
    try:
        from context.context_builder import ContextBuilder
        cb = ContextBuilder()
        ctx_result = cb.build_context(question.strip(), reranked)
        st.session_state["context"] = ctx_result
        progress.progress(37, text="✅ Context prepared")
    except Exception as e:
        errors["context"] = str(e)
        ctx_result = None
        st.warning(f"⚠️ Context builder failed: {e}")

    # ── Stage 4: Generation ────────────────────────────────────────────────
    original_answer = None
    try:
        from generation.generator import LLMGenerator
        from generation.llm_client import get_llm_client
        llm_client = get_llm_client()
        gen = LLMGenerator(client=llm_client)
        if ctx_result is None:
            raise RuntimeError("Context not available — skipping generation.")
        t0 = time.perf_counter()
        gen_result = gen.generate_answer(ctx_result)
        original_answer = gen_result.get("answer", "")
        latency["LLM Generation"] = time.perf_counter() - t0
        st.session_state["original_answer"] = original_answer
        progress.progress(50, text="✅ Answer generated")
    except RuntimeError as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            errors["generation"] = "Gemini API quota exhausted (429 RESOURCE_EXHAUSTED)."
            st.warning(
                "⚠️ **LLM generation is currently unavailable** — Gemini API quota exceeded. "
                "The retrieval and verification components can still be inspected using the retrieved evidence below."
            )
        elif "503" in err:
            errors["generation"] = "Gemini API temporarily unavailable (503)."
            st.warning("⚠️ Gemini is temporarily unavailable (503). Please retry in a moment.")
        else:
            errors["generation"] = err
            st.warning(f"⚠️ Generation failed: {err}")
        progress.progress(50, text="⚠️ Generation skipped")
    except Exception as e:
        errors["generation"] = str(e)
        st.warning(f"⚠️ Generation error: {e}")
        progress.progress(50, text="⚠️ Generation skipped")

    # ── Stage 5: Claim Extraction ──────────────────────────────────────────
    claims = []
    if original_answer:
        try:
            from claims.claim_extractor import ClaimExtractor
            extractor = ClaimExtractor()
            t0 = time.perf_counter()
            ext_result = extractor.extract_claims(original_answer, question.strip())
            claims = ext_result.get("claims", [])
            latency["Claim Extraction"] = time.perf_counter() - t0
            st.session_state["claims"] = claims
            progress.progress(62, text=f"✅ {len(claims)} claim(s) extracted")
        except Exception as e:
            errors["extraction"] = str(e)
            st.warning(f"⚠️ Claim extraction failed: {e}")
            progress.progress(62, text="⚠️ Claim extraction skipped")

    # ── Stage 6: Evidence Retrieval per Claim ─────────────────────────────
    evidence_results = []
    if claims:
        try:
            t0 = time.perf_counter()
            evidence_results = ev_retriever.retrieve_for_claims(claims)
            latency["Claim Evidence Retrieval"] = time.perf_counter() - t0
            st.session_state["evidence_results"] = evidence_results
            progress.progress(75, text="✅ Claim evidence retrieved")
        except Exception as e:
            errors["evidence"] = str(e)
            st.warning(f"⚠️ Evidence retrieval failed: {e}")
            progress.progress(75, text="⚠️ Evidence retrieval skipped")

    # ── Stage 7: NLI Verification ──────────────────────────────────────────
    verification_results = []
    if evidence_results:
        try:
            t0 = time.perf_counter()
            for ev_res in evidence_results:
                vr = verifier.verify(ev_res)
                verification_results.append(vr)
            latency["NLI Verification"] = time.perf_counter() - t0
            st.session_state["verification_results"] = verification_results
            progress.progress(87, text="✅ Claims verified")
        except Exception as e:
            errors["verification"] = str(e)
            st.warning(f"⚠️ Verification failed: {e}")
            progress.progress(87, text="⚠️ Verification skipped")

    # ── Stage 8: Correction ────────────────────────────────────────────────
    corrected_answer = None
    correction_changes = []
    if verification_results and original_answer:
        if corrector is None:
            st.info("ℹ️ Correction skipped — Gemini corrector unavailable (API key missing or quota exceeded).")
            # Derive correction actions deterministically from verification
            for vr in verification_results:
                status = vr.get("status", "UNSUPPORTED")
                action = "PRESERVED" if status == "SUPPORTED" else (
                    "CORRECTED" if status == "CONTRADICTED" else "REMOVED_OR_QUALIFIED"
                )
                correction_changes.append({"claim_id": vr["claim_id"], "action": action})
            corrected_answer = original_answer + "\n\n*(Correction unavailable — Gemini API quota exceeded.)*"
        else:
            try:
                t0 = time.perf_counter()
                corr_result = corrector.correct(
                    original_answer, claims, evidence_results, verification_results
                )
                corrected_answer = corr_result.get("corrected_answer", original_answer)
                correction_changes = corr_result.get("changes", [])
                latency["Correction"] = time.perf_counter() - t0
            except RuntimeError as e:
                err = str(e)
                if "429" in err or "quota" in err.lower():
                    st.warning("⚠️ Correction skipped — Gemini API quota exceeded.")
                else:
                    st.warning(f"⚠️ Correction failed: {err}")
                correction_changes = []
                corrected_answer = original_answer + "\n\n*(Correction unavailable.)*"

        st.session_state["corrected_answer"] = corrected_answer
        st.session_state["correction_changes"] = correction_changes
        progress.progress(95, text="✅ Answer corrected")

    # ── Stage 9: Reliability Report ────────────────────────────────────────
    if claims and evidence_results and verification_results:
        try:
            report = report_gen.generate_report(
                question=question.strip(),
                original_answer=original_answer or "",
                corrected_answer=corrected_answer or "",
                extracted_claims=claims,
                evidence_results=evidence_results,
                verification_results=verification_results,
                correction_changes=correction_changes,
            )
            st.session_state["reliability_report"] = report
        except Exception as e:
            errors["report"] = str(e)
            st.warning(f"⚠️ Reliability report failed: {e}")

    st.session_state["latency"] = latency
    st.session_state["errors"] = errors
    st.session_state["pipeline_ran"] = True
    progress.progress(100, text="✅ Pipeline complete")

elif ask_btn and not question.strip():
    st.warning("Please enter a question before running the pipeline.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Results — only shown after pipeline has run
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("pipeline_ran"):

    # ── Section 3: Original Answer ─────────────────────────────────────────
    st.header("💬 3. Original Answer")
    original_answer = st.session_state.get("original_answer")
    if original_answer:
        st.markdown(f"> {original_answer}")
    else:
        st.info(
            "LLM generation is currently unavailable. "
            "The retrieval and verification components can still be inspected below."
        )

    st.divider()

    # ── Section 4: Reliability Summary ────────────────────────────────────
    report = st.session_state.get("reliability_report")
    if report:
        st.header("📊 4. Reliability Summary")
        st.caption(
            "**Interpretation:** These percentages describe the relationship between "
            "extracted claims and the supplied evidence — they are NOT calibrated truth probabilities."
        )

        data = report.to_dict()
        metrics = data["metrics"]
        summary = summarise_metrics(metrics)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Claims", summary["total"])
        col2.metric("✅ Supported", summary["supported"])
        col3.metric("⚠️ Unsupported", summary["unsupported"])
        col4.metric("❌ Contradicted", summary["contradicted"])

        st.markdown(f"**Claim Support Rate:** {summary['support_rate']}")
        st.markdown(f"**Problematic Claim Rate:** {summary['problematic_rate']}")
        st.divider()

    # ── Section 5: Claim Verification ─────────────────────────────────────
    claims = st.session_state.get("claims", [])
    verification_results = st.session_state.get("verification_results", [])
    correction_changes = st.session_state.get("correction_changes", [])
    evidence_results = st.session_state.get("evidence_results", [])

    if report:
        st.header("🔬 5. Claim Verification & Evidence")

        data = report.to_dict()
        report_claims = data.get("claims", [])

        for rc in report_claims:
            cid = rc["claim_id"]
            status = rc["verification_status"]
            score = rc["verification_score"]
            icon = get_status_icon(status)
            corr_action = rc.get("correction_action")
            corr_label = get_correction_label(corr_action)

            with st.expander(f"{cid} — {status} {icon}", expanded=(status != "SUPPORTED")):
                st.markdown(f"**Claim:**  \n{rc['claim_text']}")
                st.markdown(f"**Verification:** `{status}` {icon}")
                st.markdown(f"**Verifier Score:** {format_score_note(score)}")
                st.markdown(f"**Correction:** {corr_label}")

                if status == "UNSUPPORTED":
                    st.info("The supplied evidence does not establish this claim. This does not necessarily mean the claim is false in the real world.")
                elif status == "CONTRADICTED":
                    st.error("The supplied evidence explicitly conflicts with this claim.")

                # ── Evidence chain ─────────────────────────────────────
                evidence = rc.get("evidence", [])
                if evidence:
                    chain = build_evidence_chain(evidence)
                    st.markdown("**Evidence Chain:**")
                    for ev in chain:
                        st.markdown(f"""
---
**{ev['evidence_id']}**  
📄 **Document:** `{ev['document']}`  
📖 **Pages:** {ev['pages']}  
🆔 **Chunk ID:** `{ev['chunk_id']}`  
> {ev['text'][:600]}{"…" if len(ev['text']) > 600 else ""}
""")
                else:
                    st.warning("No evidence retrieved for this claim.")

        st.divider()

    # ── Section 6: Corrected Answer ────────────────────────────────────────
    corrected_answer = st.session_state.get("corrected_answer")
    if corrected_answer:
        st.header("✏️ 6. Evidence-Grounded Corrected Answer")
        st.caption(
            "This answer has been revised using the verification results and supplied evidence. "
            "It is NOT guaranteed to be hallucination-free — it reflects the grounding decisions "
            "of the correction model operating on the supplied context."
        )
        st.markdown(f"> {corrected_answer}")
        st.divider()

    # ── Section 7: Retrieved Evidence (FAISS/Reranked) ────────────────────
    reranked = st.session_state.get("reranked_chunks", [])
    retrieved = st.session_state.get("retrieved_chunks", [])
    if reranked or retrieved:
        st.header("📚 7. Retrieved Evidence Chunks")
        chunks_to_show = reranked if reranked else retrieved
        label = "CrossEncoder-Reranked" if reranked else "FAISS Only"
        st.caption(f"Showing top-{len(chunks_to_show)} chunks ({label})")
        for i, chunk in enumerate(chunks_to_show, 1):
            doc = chunk.get("document_name", "")
            pg_s = chunk.get("page_start", "")
            pg_e = chunk.get("page_end", "")
            pages = f"{pg_s}–{pg_e}" if pg_e and pg_e != pg_s else str(pg_s)
            rerank_score = chunk.get("rerank_score", None)
            faiss_score = chunk.get("retrieval_score", chunk.get("score", None))

            with st.expander(f"Rank {i} — {doc} (p. {pages})"):
                st.markdown(f"> {chunk.get('text', '')[:800]}")
                cols = st.columns(2)
                if faiss_score is not None:
                    cols[0].caption(f"FAISS score: `{faiss_score:.4f}` *(not a probability)*")
                if rerank_score is not None:
                    cols[1].caption(f"CrossEncoder score: `{rerank_score:.4f}` *(not a probability)*")
        st.divider()

    # ── Section 8: Research Details ────────────────────────────────────────
    with st.expander("🔬 Research Details (Advanced)", expanded=False):
        st.subheader("Latency Measurements")
        st.caption("Local deterministic models only. Gemini latency not included if generation was skipped.")
        latency = st.session_state.get("latency", {})
        if latency:
            for stage, secs in latency.items():
                st.markdown(f"- **{stage}:** `{secs:.3f}s`")
        else:
            st.caption("No latency data available.")

        st.subheader("Errors Encountered")
        errors = st.session_state.get("errors", {})
        if errors:
            for stage, msg in errors.items():
                st.error(f"**{stage}:** {msg}")
        else:
            st.success("No errors encountered.")

        if report:
            st.subheader("Raw Reliability Report (JSON)")
            st.json(report.to_dict(), expanded=False)
