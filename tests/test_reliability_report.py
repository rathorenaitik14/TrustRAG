import pytest
from evaluation.reliability_report import ReliabilityReportGenerator
from copy import deepcopy

@pytest.fixture
def sample_inputs():
    extracted_claims = [
        {"claim_id": "C1", "claim_text": "Supported claim."},
        {"claim_id": "C2", "claim_text": "Unsupported claim."},
        {"claim_id": "C3", "claim_text": "Contradicted claim."}
    ]
    
    evidence_results = [
        {
            "claim_id": "C1",
            "evidence": [{
                "evidence_id": "C1-E1",
                "document_name": "doc1.pdf",
                "page_start": 1,
                "page_end": 2,
                "chunk_id": "chunk1",
                "text": "Supports C1."
            }]
        },
        {
            "claim_id": "C2",
            "evidence": []
        },
        {
            "claim_id": "C3",
            "evidence": [{
                "evidence_id": "C3-E1",
                "document_name": "doc1.pdf",
                "page_start": 5,
                "chunk_id": "chunk2",
                "text": "Contradicts C3."
            }]
        }
    ]
    
    verification_results = [
        {"claim_id": "C1", "status": "SUPPORTED", "score": 0.9, "reason": "A"},
        {"claim_id": "C2", "status": "UNSUPPORTED", "score": 0.5, "reason": "B"},
        {"claim_id": "C3", "status": "CONTRADICTED", "score": 0.8, "reason": "C"}
    ]
    
    correction_changes = [
        {"claim_id": "C1", "action": "PRESERVED"},
        {"claim_id": "C2", "action": "REMOVED_OR_QUALIFIED"},
        {"claim_id": "C3", "action": "CORRECTED"}
    ]
    
    return {
        "question": "Test question?",
        "original_answer": "Original.",
        "corrected_answer": "Corrected.",
        "extracted_claims": extracted_claims,
        "evidence_results": evidence_results,
        "verification_results": verification_results,
        "correction_changes": correction_changes
    }

def test_report_creation(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    data = report.to_dict()
    assert data is not None

def test_question_preservation(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    assert report.to_dict()["question"] == "Test question?"

def test_original_answer_preservation(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    assert report.to_dict()["original_answer"] == "Original."

def test_corrected_answer_preservation(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    assert report.to_dict()["corrected_answer"] == "Corrected."

def test_claim_preservation(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    claims = report.to_dict()["claims"]
    assert len(claims) == 3
    assert claims[0]["claim_text"] == "Supported claim."

def test_evidence_preservation(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    claims = report.to_dict()["claims"]
    assert len(claims[0]["evidence"]) == 1
    assert claims[0]["evidence"][0]["evidence_id"] == "C1-E1"

def test_provenance_preservation(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    claims = report.to_dict()["claims"]
    ev = claims[0]["evidence"][0]
    assert ev["document"] == "doc1.pdf"
    assert ev["pages"] == "1-2"
    assert ev["chunk_id"] == "chunk1"
    assert ev["text"] == "Supports C1."

def test_supported_count(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    assert report.to_dict()["metrics"]["supported_claims"] == 1

def test_unsupported_count(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    assert report.to_dict()["metrics"]["unsupported_claims"] == 1

def test_contradicted_count(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    assert report.to_dict()["metrics"]["contradicted_claims"] == 1

def test_total_claim_count(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    assert report.to_dict()["metrics"]["total_claims"] == 3

def test_claim_support_rate(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    # 1 supported out of 3 = 0.333...
    assert abs(report.to_dict()["metrics"]["claim_support_rate"] - 0.3333) < 0.001

def test_problematic_claim_rate(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    # 2 problematic out of 3 = 0.666...
    assert abs(report.to_dict()["metrics"]["problematic_claim_rate"] - 0.6666) < 0.001

def test_zero_claim_behavior(sample_inputs):
    sample_inputs["extracted_claims"] = []
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    metrics = report.to_dict()["metrics"]
    assert metrics["total_claims"] == 0
    assert metrics["claim_support_rate"] == 0.0
    assert metrics["problematic_claim_rate"] == 0.0

def test_correction_action_counts(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    metrics = report.to_dict()["metrics"]
    assert metrics["preserved_claims"] == 1
    assert metrics["corrected_claims"] == 1
    assert metrics["removed_or_qualified_claims"] == 1

def test_human_readable_output(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    text = report.to_text()
    assert "TRUSTRAG RELIABILITY REPORT" in text
    assert "Original." in text
    assert "Corrected." in text
    assert "Total Claims: 3" in text

def test_json_output(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    j = report.to_json()
    assert '"total_claims": 3' in j

def test_deterministic_report_generation(sample_inputs):
    generator = ReliabilityReportGenerator()
    r1 = generator.generate_report(**sample_inputs).to_text()
    r2 = generator.generate_report(**sample_inputs).to_text()
    assert r1 == r2

def test_unsupported_not_treated_as_contradicted(sample_inputs):
    sample_inputs["verification_results"][1]["status"] = "UNSUPPORTED"
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    assert report.to_dict()["metrics"]["contradicted_claims"] == 1 # still 1
    assert report.to_dict()["metrics"]["unsupported_claims"] == 1

def test_contradicted_not_treated_as_unsupported(sample_inputs):
    sample_inputs["verification_results"][2]["status"] = "CONTRADICTED"
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    assert report.to_dict()["metrics"]["contradicted_claims"] == 1
    assert report.to_dict()["metrics"]["unsupported_claims"] == 1

def test_no_verification_status_is_changed(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    claims = report.to_dict()["claims"]
    assert claims[0]["verification_status"] == "SUPPORTED"
    assert claims[1]["verification_status"] == "UNSUPPORTED"
    assert claims[2]["verification_status"] == "CONTRADICTED"

def test_no_source_metadata_is_lost(sample_inputs):
    generator = ReliabilityReportGenerator()
    report = generator.generate_report(**sample_inputs)
    c3_ev = report.to_dict()["claims"][2]["evidence"][0]
    assert c3_ev["pages"] == "5" # page_end was empty/None

def test_original_input_objects_not_mutated(sample_inputs):
    inputs_copy = deepcopy(sample_inputs)
    generator = ReliabilityReportGenerator()
    generator.generate_report(**sample_inputs)
    
    assert sample_inputs["extracted_claims"] == inputs_copy["extracted_claims"]
    assert sample_inputs["evidence_results"] == inputs_copy["evidence_results"]
    assert sample_inputs["verification_results"] == inputs_copy["verification_results"]
    assert sample_inputs["correction_changes"] == inputs_copy["correction_changes"]
