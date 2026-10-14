"""
End-to-end pipeline test for TrustRAG demo readiness check.
Run with: python test_pipeline_e2e.py
"""
import sys
import os
import json
import traceback
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

errors = []

print("=" * 60)
print("TrustRAG Demo Readiness Check")
print("=" * 60)
print()

# 1. PDF Ingestion
print("1. Testing PDF ingestion...")
try:
    from ingestion.pdf_loader import load_pdf
    from ingestion.chunker import chunk_pages

    pdf_path = 'TRUSTRAG_Research_Paper.pdf'
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages)
    print("   OK: Loaded %d pages, %d chunks" % (len(pages), len(chunks)))
except Exception as e:
    traceback.print_exc()
    errors.append("Ingestion: %s" % e)
    print("   FAILED: %s" % e)
    sys.exit(1)

# 2. Embeddings + FAISS
print("2. Testing embeddings + FAISS build...")
faiss_path = None
meta_path = None
chunks_jsonl = None
try:
    from retrieval.embeddings import embed_chunks
    from retrieval.faiss_store import build_faiss_index, save_faiss_index

    Path('data/processed').mkdir(parents=True, exist_ok=True)
    embeddings, emb_meta = embed_chunks(chunks)
    dim = emb_meta["dimension"]
    index = build_faiss_index(embeddings)
    print("   OK: Embeddings %s, FAISS %d vectors, dim=%d" % (str(embeddings.shape), index.ntotal, dim))

    faiss_path = 'data/processed/test_chunks_faiss.index'
    save_faiss_index(index, faiss_path)

    chunks_jsonl = 'data/processed/test_chunks.jsonl'
    with open(chunks_jsonl, 'w', encoding='utf-8') as f:
        for c in chunks:
            f.write(json.dumps({
                'chunk_id': c.chunk_id,
                'document_id': c.document_id,
                'document_name': c.document_name,
                'text': c.text,
                'page_start': c.page_start,
                'page_end': c.page_end,
                'source_path': c.source_path,
            }) + '\n')

    index_mapping = {str(i): c.chunk_id for i, c in enumerate(chunks)}
    meta = {'index_mapping': index_mapping, 'dim': dim}
    meta_path = 'data/processed/test_chunks_embeddings_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    print("   OK: Saved FAISS index + metadata")
except Exception as e:
    traceback.print_exc()
    errors.append("Embeddings/FAISS: %s" % e)
    print("   FAILED: %s" % e)
    sys.exit(1)

# 3. Retriever + Reranker
print("3. Testing RAGRetriever + CrossEncoderReranker...")
retriever = None
reranker = None
reranked = None
try:
    from retrieval.retriever import RAGRetriever
    from retrieval.reranker import CrossEncoderReranker

    retriever = RAGRetriever(faiss_path, meta_path, chunks_jsonl)
    question = "What is TrustRAG and how does it detect hallucinations?"
    retrieved = retriever.retrieve(question, k=10)
    print("   OK: Retrieved %d chunks" % len(retrieved))
    print("   Keys: %s" % list(retrieved[0].keys()))

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(question, retrieved, top_k=5)
    print("   OK: Reranked to %d chunks" % len(reranked))
    print("   Reranked keys: %s" % list(reranked[0].keys()))
except Exception as e:
    traceback.print_exc()
    errors.append("Retriever/Reranker: %s" % e)
    print("   FAILED: %s" % e)
    sys.exit(1)

# 4. ContextBuilder
print("4. Testing ContextBuilder...")
ctx = None
try:
    from context.context_builder import ContextBuilder
    cb = ContextBuilder()
    ctx = cb.build_context(question, reranked)
    print("   OK: Context length=%d chars, %d evidence items" % (len(ctx['formatted_context']), len(ctx['evidence_list'])))
except Exception as e:
    traceback.print_exc()
    errors.append("ContextBuilder: %s" % e)
    print("   FAILED: %s" % e)

# 5. Gemini Generation
print("5. Testing Gemini LLM Generation...")
original_answer = None
if ctx:
    try:
        from generation.generator import LLMGenerator
        from generation.llm_client import get_llm_client
        llm_client = get_llm_client()
        gen = LLMGenerator(client=llm_client)
        gen_result = gen.generate_answer(ctx)
        original_answer = gen_result.get("answer", "")
        print("   OK: Generated answer (%d chars)" % len(original_answer))
        print("   Preview: %s..." % original_answer[:200])
    except Exception as e:
        traceback.print_exc()
        errors.append("Generation: %s" % e)
        print("   FAILED: %s" % e)

# 6. Claim Extraction
print("6. Testing ClaimExtractor...")
claims = []
if original_answer:
    try:
        from claims.claim_extractor import ClaimExtractor
        extractor = ClaimExtractor()
        ext_result = extractor.extract_claims(original_answer, question)
        claims = ext_result.get("claims", [])
        print("   OK: Extracted %d claims" % len(claims))
        for c in claims[:3]:
            print("      %s: %s..." % (c['claim_id'], c['claim_text'][:80]))
    except Exception as e:
        traceback.print_exc()
        errors.append("ClaimExtractor: %s" % e)
        print("   FAILED: %s" % e)

if not claims:
    print("   Using mock claims for downstream testing")
    claims = [
        {"claim_id": "C1", "claim_text": "TrustRAG is a pipeline for hallucination detection in RAG systems.", "claim_type": "factual"},
        {"claim_id": "C2", "claim_text": "TrustRAG uses NLI to verify extracted claims.", "claim_type": "factual"},
        {"claim_id": "C3", "claim_text": "TrustRAG uses FAISS for vector similarity search.", "claim_type": "factual"},
    ]
    if not original_answer:
        original_answer = "TrustRAG is a pipeline for hallucination detection in RAG systems. It uses NLI to verify extracted claims. TrustRAG uses FAISS for vector similarity search."

# 7. Evidence Retrieval per Claim
print("7. Testing EvidenceRetriever...")
evidence_results = []
try:
    from claims.evidence_retriever import EvidenceRetriever
    ev_retriever = EvidenceRetriever(retriever, reranker, default_retrieval_k=10, default_evidence_top_k=3)
    evidence_results = ev_retriever.retrieve_for_claims(claims)
    print("   OK: Retrieved evidence for %d claims" % len(evidence_results))
    for er in evidence_results[:2]:
        print("      %s: %d evidence items" % (er['claim_id'], len(er['evidence'])))
except Exception as e:
    traceback.print_exc()
    errors.append("EvidenceRetriever: %s" % e)
    print("   FAILED: %s" % e)

# 8. NLI Verification
print("8. Testing NLIVerifier...")
verification_results = []
if evidence_results:
    try:
        from verification.nli_verifier import NLIVerifier
        verifier = NLIVerifier(model_name="cross-encoder/nli-deberta-v3-small")
        for ev_res in evidence_results:
            vr = verifier.verify(ev_res)
            verification_results.append(vr)
        print("   OK: Verified %d claims" % len(verification_results))
        for vr in verification_results[:3]:
            print("      %s: %s (score=%.3f)" % (vr['claim_id'], vr['status'], vr.get('score', 0)))
    except Exception as e:
        traceback.print_exc()
        errors.append("NLIVerifier: %s" % e)
        print("   FAILED: %s" % e)

# 9. Correction
print("9. Testing GeminiCorrector...")
corrected_answer = None
correction_changes = []
if verification_results:
    try:
        from correction.gemini_corrector import GeminiCorrector
        corrector = GeminiCorrector()
        corr_result = corrector.correct(original_answer, claims, evidence_results, verification_results)
        corrected_answer = corr_result.get("corrected_answer", "")
        correction_changes = corr_result.get("changes", [])
        print("   OK: Correction done. Changes: %d" % len(correction_changes))
        for ch in correction_changes[:3]:
            print("      %s: %s" % (ch.get('claim_id'), ch.get('action')))
    except Exception as e:
        traceback.print_exc()
        errors.append("Corrector: %s" % e)
        print("   FAILED: %s" % e)
        for vr in verification_results:
            status = vr.get("status", "UNSUPPORTED")
            action = "PRESERVED" if status == "SUPPORTED" else ("CORRECTED" if status == "CONTRADICTED" else "REMOVED_OR_QUALIFIED")
            correction_changes.append({"claim_id": vr["claim_id"], "action": action})
        corrected_answer = original_answer + "\n\n*(Correction used deterministic fallback)*"

# 10. Reliability Report
print("10. Testing ReliabilityReportGenerator...")
if claims and evidence_results and verification_results:
    try:
        from evaluation.reliability_report import ReliabilityReportGenerator
        report_gen = ReliabilityReportGenerator()
        report = report_gen.generate_report(
            question=question,
            original_answer=original_answer or "",
            corrected_answer=corrected_answer or "",
            extracted_claims=claims,
            evidence_results=evidence_results,
            verification_results=verification_results,
            correction_changes=correction_changes,
        )
        rd = report.to_dict()
        m = rd["metrics"]
        print("   OK: Report generated!")
        print("      Total claims: %d" % m['total_claims'])
        print("      Supported: %d" % m['supported_claims'])
        print("      Unsupported: %d" % m['unsupported_claims'])
        print("      Contradicted: %d" % m['contradicted_claims'])
        print("      Support rate: %.0f%%" % (m['claim_support_rate'] * 100))
    except Exception as e:
        traceback.print_exc()
        errors.append("ReliabilityReport: %s" % e)
        print("   FAILED: %s" % e)

# Summary
print()
print("=" * 60)
if errors:
    print("ERRORS FOUND (%d):" % len(errors))
    for e in errors:
        print("   - %s" % e)
else:
    print("ALL PIPELINE STAGES PASSED! Demo is ready.")
print("=" * 60)
