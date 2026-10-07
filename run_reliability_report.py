import sys
from evaluation.reliability_report import ReliabilityReportGenerator

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    question = "What is the goal of data mining, and when was it invented?"
    original_answer = "The goal of data mining is to provide efficient tools and techniques for KDD. Data mining was invented in 1970."
    corrected_answer = "The goal of data mining is to provide efficient tools and techniques for KDD. The provided evidence does not establish when data mining was invented."
    
    extracted_claims = [
        {"claim_id": "C1", "claim_text": "The goal of data mining is to provide efficient tools and techniques for KDD."},
        {"claim_id": "C2", "claim_text": "Data mining was invented in 1970."}
    ]
    
    evidence_results = [
        {
            "claim_id": "C1",
            "evidence": [{
                "evidence_id": "C1-E1",
                "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
                "page_start": 8,
                "page_end": 18,
                "chunk_id": "chunk_123",
                "text": "Goal of Data Mining is to provide efficient tools and techniques for KDD"
            }]
        },
        {
            "claim_id": "C2",
            "evidence": [{
                "evidence_id": "C2-E1",
                "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
                "page_start": 8,
                "page_end": 18,
                "chunk_id": "chunk_124",
                "text": "1970s: Relational data model... 1990s: Data mining, data warehousing..."
            }]
        }
    ]
    
    verification_results = [
        {"claim_id": "C1", "status": "SUPPORTED", "score": 0.98, "reason": "The strongest relevant evidence supports the claim."},
        {"claim_id": "C2", "status": "UNSUPPORTED", "score": 0.0, "reason": "Evidence is insufficient to establish either entailment or contradiction."}
    ]
    
    correction_changes = [
        {"claim_id": "C1", "action": "PRESERVED"},
        {"claim_id": "C2", "action": "REMOVED_OR_QUALIFIED"}
    ]
    
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(
        question=question,
        original_answer=original_answer,
        corrected_answer=corrected_answer,
        extracted_claims=extracted_claims,
        evidence_results=evidence_results,
        verification_results=verification_results,
        correction_changes=correction_changes
    )
    
    print(report.to_text())

if __name__ == "__main__":
    main()
