import json
from typing import List, Dict, Any, Optional
from .models import ReliabilityReportDict, ReportMetrics, ReportClaim, ReportEvidence

class ReliabilityReportGenerator:
    """
    Aggregates TRUSTRAG pipeline outputs into a comprehensive Reliability Report.
    """
    def generate_report(
        self,
        question: str,
        original_answer: str,
        corrected_answer: str,
        extracted_claims: List[Dict[str, Any]],
        evidence_results: List[Dict[str, Any]],
        verification_results: List[Dict[str, Any]],
        correction_changes: List[Dict[str, Any]]
    ) -> 'ReliabilityReport':
        
        total_claims = len(extracted_claims)
        supported_claims = 0
        unsupported_claims = 0
        contradicted_claims = 0
        
        preserved_claims = 0
        corrected_claims = 0
        removed_or_qualified_claims = 0
        
        report_claims: List[ReportClaim] = []
        
        # Build maps for easy lookup
        ev_map = {cr["claim_id"]: cr["evidence"] for cr in evidence_results}
        vr_map = {vr["claim_id"]: vr for vr in verification_results}
        corr_map = {c["claim_id"]: c for c in correction_changes}
        
        for claim in extracted_claims:
            c_id = claim.get("claim_id", "")
            c_text = claim.get("claim_text", "")
            
            # Verifications
            vr = vr_map.get(c_id, {})
            v_status = vr.get("status", "UNSUPPORTED")
            v_score = vr.get("score", 0.0)
            v_reason = vr.get("reason")
            
            if v_status == "SUPPORTED":
                supported_claims += 1
            elif v_status == "CONTRADICTED":
                contradicted_claims += 1
            else:
                unsupported_claims += 1
                
            # Corrections
            corr = corr_map.get(c_id, {})
            c_action = corr.get("action")
            
            if c_action == "PRESERVED":
                preserved_claims += 1
            elif c_action == "CORRECTED":
                corrected_claims += 1
            elif c_action == "REMOVED_OR_QUALIFIED":
                removed_or_qualified_claims += 1
                
            # Evidence
            ev_list = ev_map.get(c_id, [])
            report_evidences: List[ReportEvidence] = []
            ev_ids = []
            
            # Map evidence
            for ev in ev_list:
                # We want to preserve exact provenance
                page_s = ev.get("page_start", "")
                page_e = ev.get("page_end", "")
                pages = str(page_s)
                if page_e and page_e != page_s:
                    pages += f"-{page_e}"
                    
                report_evidences.append({
                    "evidence_id": ev.get("evidence_id", ""),
                    "document": ev.get("document_name", ""),
                    "pages": pages,
                    "chunk_id": ev.get("chunk_id", ""),
                    "text": ev.get("text", "")
                })
                ev_ids.append(ev.get("evidence_id", ""))
            
            # Filter evidence_ids to those actually evaluated positively?
            # User requirement: preserve ALL evidence. 
            
            report_claims.append({
                "claim_id": c_id,
                "claim_text": c_text,
                "verification_status": v_status,
                "verification_score": v_score,
                "evidence_ids": ev_ids,
                "evidence": report_evidences,
                "correction_action": c_action,
                "correction_status_reason": v_reason
            })
            
        claim_support_rate = (supported_claims / total_claims) if total_claims > 0 else 0.0
        problematic_claim_rate = ((unsupported_claims + contradicted_claims) / total_claims) if total_claims > 0 else 0.0
        
        metrics: ReportMetrics = {
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "contradicted_claims": contradicted_claims,
            "preserved_claims": preserved_claims,
            "corrected_claims": corrected_claims,
            "removed_or_qualified_claims": removed_or_qualified_claims,
            "claim_support_rate": float(claim_support_rate),
            "problematic_claim_rate": float(problematic_claim_rate)
        }
        
        summary = (
            f"Total claims: {total_claims}\n"
            f"Supported: {supported_claims}\n"
            f"Unsupported: {unsupported_claims}\n"
            f"Contradicted: {contradicted_claims}\n\n"
            f"Claim support rate: {claim_support_rate * 100:.0f}%\n"
            f"Problematic claim rate: {problematic_claim_rate * 100:.0f}%\n\n"
            f"Preserved: {preserved_claims}\n"
            f"Corrected: {corrected_claims}\n"
            f"Removed/qualified: {removed_or_qualified_claims}"
        )
        
        report_dict: ReliabilityReportDict = {
            "question": question,
            "original_answer": original_answer,
            "corrected_answer": corrected_answer,
            "claims": report_claims,
            "metrics": metrics,
            "summary": summary
        }
        
        return ReliabilityReport(report_dict)

class ReliabilityReport:
    def __init__(self, data: ReliabilityReportDict):
        self.data = data
        
    def to_dict(self) -> ReliabilityReportDict:
        return self.data
        
    def to_json(self, indent=2) -> str:
        return json.dumps(self.data, indent=indent)
        
    def to_text(self) -> str:
        """
        Returns a deterministic human-readable text representation.
        """
        lines = []
        lines.append("==================================================")
        lines.append("TRUSTRAG RELIABILITY REPORT")
        lines.append("==================================================")
        lines.append("")
        lines.append("QUESTION:")
        lines.append(self.data["question"])
        lines.append("")
        lines.append("ORIGINAL ANSWER:")
        lines.append(self.data["original_answer"])
        lines.append("")
        lines.append("CLAIMS")
        lines.append("--------------------------------------------------")
        
        for c in self.data["claims"]:
            lines.append("")
            lines.append(c["claim_id"])
            lines.append("Claim:")
            lines.append(c["claim_text"])
            lines.append("")
            lines.append("Status:")
            lines.append(c["verification_status"])
            lines.append("")
            lines.append("Verification Score:")
            lines.append(f"{c['verification_score']:.4f}")
            lines.append("")
            
            if c["evidence"]:
                lines.append("Evidence:")
                for ev in c["evidence"]:
                    lines.append(ev["evidence_id"])
                    lines.append(f"Document: {ev['document']}")
                    lines.append(f"Pages: {ev['pages']}")
                    lines.append(f"Chunk ID: {ev['chunk_id']}")
                    lines.append(f"Text: {ev['text']}")
                    lines.append("")
            else:
                lines.append("Evidence:")
                lines.append("None")
                lines.append("")
                
            lines.append("Correction:")
            lines.append(str(c["correction_action"]))
            lines.append("")
            lines.append("--------------------------------------------------")
            
        lines.append("")
        lines.append("SUMMARY")
        lines.append("--------------------------------------------------")
        lines.append("")
        lines.append(f"Total Claims: {self.data['metrics']['total_claims']}")
        lines.append("")
        lines.append(f"Supported: {self.data['metrics']['supported_claims']}")
        lines.append("")
        lines.append(f"Unsupported: {self.data['metrics']['unsupported_claims']}")
        lines.append("")
        lines.append(f"Contradicted: {self.data['metrics']['contradicted_claims']}")
        lines.append("")
        lines.append(f"Claim Support Rate: {self.data['metrics']['claim_support_rate']*100:.0f}%")
        lines.append("")
        lines.append(f"Problematic Claim Rate: {self.data['metrics']['problematic_claim_rate']*100:.0f}%")
        lines.append("")
        lines.append(f"Preserved: {self.data['metrics']['preserved_claims']}")
        lines.append("")
        lines.append(f"Corrected: {self.data['metrics']['corrected_claims']}")
        lines.append("")
        lines.append(f"Removed/Qualified: {self.data['metrics']['removed_or_qualified_claims']}")
        lines.append("")
        lines.append("==================================================")
        lines.append("")
        lines.append("CORRECTED ANSWER")
        lines.append("==================================================")
        lines.append("")
        lines.append(self.data["corrected_answer"])
        
        return "\n".join(lines)
