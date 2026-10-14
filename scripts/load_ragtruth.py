import json
import random
from pathlib import Path
from datasets import load_dataset

# Setup paths
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
out_path = data_dir / "ragtruth_test.json"

print("Loading wandb/RAGTruth-processed...")
ds = load_dataset("wandb/RAGTruth-processed")

# We use the train split as it has the most examples
dataset = ds["train"]

supported = []
unsupported = []
contradicted = []

print("Categorizing examples by hallucination type...")
for item in dataset:
    labels = item.get("hallucination_labels_processed", {})
    is_conflict = labels.get("evident_conflict", 0) > 0
    is_baseless = labels.get("baseless_info", 0) > 0
    
    # Map to our labels
    if is_conflict:
        contradicted.append(item)
    elif is_baseless:
        unsupported.append(item)
    else:
        supported.append(item)

print(f"Total available: SUPPORTED={len(supported)}, UNSUPPORTED={len(unsupported)}, CONTRADICTED={len(contradicted)}")

random.seed(42)

# Sample 500 cases stratified (30% supported, 40% unsupported, 30% contradicted)
# i.e., 150, 200, 150
samp_supported = random.sample(supported, min(150, len(supported)))
samp_unsupported = random.sample(unsupported, min(200, len(unsupported)))
samp_contradicted = random.sample(contradicted, min(150, len(contradicted)))

final_sample = samp_supported + samp_unsupported + samp_contradicted
random.shuffle(final_sample)

# Format for TrustRAG pipeline
formatted_cases = []
for item in final_sample:
    labels = item.get("hallucination_labels_processed", {})
    is_conflict = labels.get("evident_conflict", 0) > 0
    is_baseless = labels.get("baseless_info", 0) > 0
    
    if is_conflict:
        gt_label = "CONTRADICTED"
    elif is_baseless:
        gt_label = "UNSUPPORTED"
    else:
        gt_label = "SUPPORTED"
        
    formatted_cases.append({
        "id": item["id"],
        "question": item["query"],
        "context": item["context"],
        "answer": item["output"],
        "task_type": item["task_type"],
        "model": item["model"],
        "ground_truth_label": gt_label,
        "hallucination_labels": item.get("hallucination_labels")
    })

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(formatted_cases, f, indent=2, ensure_ascii=False)

print(f"Successfully sampled {len(formatted_cases)} cases and saved to {out_path}.")
