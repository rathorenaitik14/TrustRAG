from typing import List
from .models import BenchmarkCase

def get_benchmark_dataset() -> List[BenchmarkCase]:
    return [
        # ==========================================
        # SUPPORTED
        # ==========================================
        {
            "case_id": "TC-SUP-1",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "What is the goal of data mining?",
            "grounded_answer": "The goal of data mining is to provide efficient tools and techniques for KDD.",
            "injected_answer": "The goal of data mining is to provide efficient tools and techniques for KDD.",
            "original_claims": ["The goal of data mining is to provide efficient tools and techniques for KDD."],
            "injected_claim": "",
            "ground_truth_label": "SUPPORTED",
            "evidence_reference": "Goal of Data Mining is to provide efficient tools and techniques for KDD",
            "expected_correction_action": "PRESERVED",
            "difficulty": "easy",
            "notes": "Direct factual extraction."
        },
        {
            "case_id": "TC-SUP-2",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "How much data will be stored in the cloud by 2025?",
            "grounded_answer": "By 2025, 49 percent of data will be stored in public cloud environments.",
            "injected_answer": "By 2025, 49 percent of data will be stored in public cloud environments.",
            "original_claims": ["By 2025, 49 percent of data will be stored in public cloud environments."],
            "injected_claim": "",
            "ground_truth_label": "SUPPORTED",
            "evidence_reference": "By 2025, 49 percent of data will be stored in public cloud environments.",
            "expected_correction_action": "PRESERVED",
            "difficulty": "easy",
            "notes": "Direct factual extraction."
        },
        {
            "case_id": "TC-SUP-3",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "How much of the work in a data mining application is data preparation?",
            "grounded_answer": "Data preparation, cleaning, and transformation comprises the majority of the work in a data mining application (90%).",
            "injected_answer": "Data preparation, cleaning, and transformation comprises the majority of the work in a data mining application (90%).",
            "original_claims": ["Data preparation, cleaning, and transformation comprises the majority of the work in a data mining application (90%)."],
            "injected_claim": "",
            "ground_truth_label": "SUPPORTED",
            "evidence_reference": "Data preparation, cleaning, and transformation comprises the majority of the work in a data mining application (90%).",
            "expected_correction_action": "PRESERVED",
            "difficulty": "easy",
            "notes": "Direct factual extraction."
        },
        {
            "case_id": "TC-SUP-4",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "How much data will IoT devices create by 2025?",
            "grounded_answer": "90ZB of data will be created on IoT devices by 2025.",
            "injected_answer": "90ZB of data will be created on IoT devices by 2025.",
            "original_claims": ["90ZB of data will be created on IoT devices by 2025."],
            "injected_claim": "",
            "ground_truth_label": "SUPPORTED",
            "evidence_reference": "90ZB of data will be created on IoT devices by 2025.",
            "expected_correction_action": "PRESERVED",
            "difficulty": "easy",
            "notes": "Direct factual extraction."
        },
        {
            "case_id": "TC-SUP-5",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "What is the relationship between quality data and mining results?",
            "grounded_answer": "No quality data means no quality mining results.",
            "injected_answer": "No quality data means no quality mining results.",
            "original_claims": ["No quality data means no quality mining results."],
            "injected_claim": "",
            "ground_truth_label": "SUPPORTED",
            "evidence_reference": "No quality data, no quality mining results!",
            "expected_correction_action": "PRESERVED",
            "difficulty": "easy",
            "notes": "Paraphrased extraction."
        },

        # ==========================================
        # UNSUPPORTED
        # ==========================================
        {
            "case_id": "TC-UNS-1",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "What is the goal of data mining?",
            "grounded_answer": "The goal of data mining is to provide efficient tools and techniques for KDD.",
            "injected_answer": "The goal of data mining is to provide efficient tools and techniques for KDD. Data mining was invented at MIT.",
            "original_claims": ["The goal of data mining is to provide efficient tools and techniques for KDD."],
            "injected_claim": "Data mining was invented at MIT.",
            "ground_truth_label": "UNSUPPORTED",
            "evidence_reference": "Goal of Data Mining is to provide efficient tools and techniques for KDD",
            "expected_correction_action": "REMOVED_OR_QUALIFIED",
            "difficulty": "medium",
            "notes": "Claim lacks any source evidence."
        },
        {
            "case_id": "TC-UNS-2",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "How much data will be stored in the cloud by 2025?",
            "grounded_answer": "By 2025, 49 percent of data will be stored in public cloud environments.",
            "injected_answer": "By 2025, 49 percent of data will be stored in public cloud environments. Public cloud environments are perfectly secure.",
            "original_claims": ["By 2025, 49 percent of data will be stored in public cloud environments."],
            "injected_claim": "Public cloud environments are perfectly secure.",
            "ground_truth_label": "UNSUPPORTED",
            "evidence_reference": "By 2025, 49 percent of data will be stored in public cloud environments.",
            "expected_correction_action": "REMOVED_OR_QUALIFIED",
            "difficulty": "medium",
            "notes": "Claim lacks any source evidence."
        },
        {
            "case_id": "TC-UNS-3",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "How much of the work in a data mining application is data preparation?",
            "grounded_answer": "Data preparation, cleaning, and transformation comprises the majority of the work in a data mining application (90%).",
            "injected_answer": "Data preparation, cleaning, and transformation comprises the majority of the work in a data mining application (90%). Data preparation should only be done using Apache Spark.",
            "original_claims": ["Data preparation, cleaning, and transformation comprises the majority of the work in a data mining application (90%)."],
            "injected_claim": "Data preparation should only be done using Apache Spark.",
            "ground_truth_label": "UNSUPPORTED",
            "evidence_reference": "Data preparation, cleaning, and transformation comprises the majority of the work in a data mining application (90%).",
            "expected_correction_action": "REMOVED_OR_QUALIFIED",
            "difficulty": "medium",
            "notes": "Tool recommendation not found in text."
        },
        {
            "case_id": "TC-UNS-4",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "How much data will IoT devices create by 2025?",
            "grounded_answer": "90ZB of data will be created on IoT devices by 2025.",
            "injected_answer": "90ZB of data will be created on IoT devices by 2025. IoT devices are the leading cause of climate change.",
            "original_claims": ["90ZB of data will be created on IoT devices by 2025."],
            "injected_claim": "IoT devices are the leading cause of climate change.",
            "ground_truth_label": "UNSUPPORTED",
            "evidence_reference": "90ZB of data will be created on IoT devices by 2025.",
            "expected_correction_action": "REMOVED_OR_QUALIFIED",
            "difficulty": "medium",
            "notes": "Opinion/external fact not found in text."
        },
        {
            "case_id": "TC-UNS-5",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "What is the relationship between quality data and mining results?",
            "grounded_answer": "No quality data means no quality mining results.",
            "injected_answer": "No quality data means no quality mining results. The best programming language for data mining is Python.",
            "original_claims": ["No quality data means no quality mining results."],
            "injected_claim": "The best programming language for data mining is Python.",
            "ground_truth_label": "UNSUPPORTED",
            "evidence_reference": "No quality data, no quality mining results!",
            "expected_correction_action": "REMOVED_OR_QUALIFIED",
            "difficulty": "medium",
            "notes": "Tool recommendation not found in text."
        },

        # ==========================================
        # CONTRADICTED
        # ==========================================
        {
            "case_id": "TC-CON-1",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "What is the relationship between quality data and mining results?",
            "grounded_answer": "No quality data means no quality mining results.",
            "injected_answer": "High quality mining results can be easily obtained even without quality data.",
            "original_claims": ["No quality data means no quality mining results."],
            "injected_claim": "High quality mining results can be easily obtained even without quality data.",
            "ground_truth_label": "CONTRADICTED",
            "evidence_reference": "No quality data, no quality mining results!",
            "expected_correction_action": "CORRECTED",
            "difficulty": "hard",
            "notes": "Directly opposes explicit source text."
        },
        {
            "case_id": "TC-CON-2",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "How much data will be stored in the cloud by 2025?",
            "grounded_answer": "By 2025, 49 percent of data will be stored in public cloud environments.",
            "injected_answer": "By 2025, exactly 0 percent of data will be stored in public cloud environments.",
            "original_claims": ["By 2025, 49 percent of data will be stored in public cloud environments."],
            "injected_claim": "By 2025, exactly 0 percent of data will be stored in public cloud environments.",
            "ground_truth_label": "CONTRADICTED",
            "evidence_reference": "By 2025, 49 percent of data will be stored in public cloud environments.",
            "expected_correction_action": "CORRECTED",
            "difficulty": "hard",
            "notes": "Contradicts explicit numerical percentage in source text."
        },
        {
            "case_id": "TC-CON-3",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "How much of the work in a data mining application is data preparation?",
            "grounded_answer": "Data preparation, cleaning, and transformation comprises the majority of the work in a data mining application (90%).",
            "injected_answer": "Data preparation comprises less than 5% of the work in data mining.",
            "original_claims": ["Data preparation, cleaning, and transformation comprises the majority of the work in a data mining application (90%)."],
            "injected_claim": "Data preparation comprises less than 5% of the work in data mining.",
            "ground_truth_label": "CONTRADICTED",
            "evidence_reference": "Data preparation, cleaning, and transformation comprises the majority of the work in a data mining application (90%).",
            "expected_correction_action": "CORRECTED",
            "difficulty": "hard",
            "notes": "Contradicts explicit majority percentage."
        },
        {
            "case_id": "TC-CON-4",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "How much data will IoT devices create by 2025?",
            "grounded_answer": "90ZB of data will be created on IoT devices by 2025.",
            "injected_answer": "IoT devices will create exactly 1 byte of data by 2025.",
            "original_claims": ["90ZB of data will be created on IoT devices by 2025."],
            "injected_claim": "IoT devices will create exactly 1 byte of data by 2025.",
            "ground_truth_label": "CONTRADICTED",
            "evidence_reference": "90ZB of data will be created on IoT devices by 2025.",
            "expected_correction_action": "CORRECTED",
            "difficulty": "hard",
            "notes": "Contradicts magnitude explicit in source."
        },
        {
            "case_id": "TC-CON-5",
            "document_id": "DEDV Unit1",
            "document_name": "DEDV Unit1_Data Preprocessing An Overview.pdf",
            "question": "What role does data mining play in the knowledge discovery process?",
            "grounded_answer": "Data mining plays an essential role in the knowledge discovery process.",
            "injected_answer": "Data mining plays no role whatsoever in the knowledge discovery process.",
            "original_claims": ["Data mining plays an essential role in the knowledge discovery process."],
            "injected_claim": "Data mining plays no role whatsoever in the knowledge discovery process.",
            "ground_truth_label": "CONTRADICTED",
            "evidence_reference": "Data mining plays an essential role in the knowledge discovery process",
            "expected_correction_action": "CORRECTED",
            "difficulty": "hard",
            "notes": "Direct semantic opposition."
        }
    ]
