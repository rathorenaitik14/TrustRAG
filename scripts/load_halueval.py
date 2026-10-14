import json
import random
from pathlib import Path
from datasets import load_dataset

# Setup paths
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
out_path = data_dir / "halueval_test.json"

print("Loading HaluEval QA dataset...")
try:
    ds = load_dataset("pminervini/HaluEval", "qa")
    split_name = "data" if "data" in ds else list(ds.keys())[0]
    raw_dataset = list(ds[split_name])
except Exception as e:
    print("Could not load pminervini/HaluEval, trying alternative...", e)
    try:
        ds = load_dataset("DongqiKuang/HaluEval", name="qa")
        split_name = "data" if "data" in ds else list(ds.keys())[0]
        raw_dataset = list(ds[split_name])
    except Exception as e2:
        print("Fallback also failed:", e2)
        raw_dataset = []

if not raw_dataset:
    print("Error: Could not load HaluEval records.")
    formatted_cases = []
else:
    random.seed(42)
    # We sample 150 items to produce 150 SUPPORTED and 150 UNSUPPORTED = 300 total cases
    target_count = 150
    selected = random.sample(raw_dataset, min(target_count, len(raw_dataset)))
    
    formatted_cases = []
    for idx, item in enumerate(selected):
        knowledge = item.get("knowledge", "")
        question = item.get("question", item.get("query", ""))
        right_ans = item.get("right_answer", "")
        halluc_ans = item.get("hallucinated_answer", "")
        
        # 1. Grounded / Supported case
        formatted_cases.append({
            "id": f"halueval-s-{idx:04d}",
            "case_id": f"halueval-s-{idx:04d}",
            "source_dataset": "halueval",
            "task_type": "qa",
            "question": question,
            "context": knowledge,
            "answer": right_ans,
            "ground_truth_label": "SUPPORTED",
            "grounded_answer": right_ans,
            "injected_answer": right_ans,
            "document_id": f"halueval-doc-{idx}",
            "document_name": f"HaluEval knowledge {idx}",
            "expected_correction_action": "PRESERVED"
        })
        
        # 2. Hallucinated / Unsupported case
        formatted_cases.append({
            "id": f"halueval-h-{idx:04d}",
            "case_id": f"halueval-h-{idx:04d}",
            "source_dataset": "halueval",
            "task_type": "qa",
            "question": question,
            "context": knowledge,
            "answer": halluc_ans,
            "ground_truth_label": "UNSUPPORTED",
            "grounded_answer": right_ans,
            "injected_answer": halluc_ans,
            "injected_claim": halluc_ans,
            "document_id": f"halueval-doc-{idx}",
            "document_name": f"HaluEval knowledge {idx}",
            "expected_correction_action": "REMOVED_OR_QUALIFIED"
        })
        
    random.shuffle(formatted_cases)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(formatted_cases, f, indent=2, ensure_ascii=False)

print(f"Successfully sampled {len(formatted_cases)} HaluEval cases and saved to {out_path}.")

