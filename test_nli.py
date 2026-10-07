from transformers import pipeline

model_name = "cross-encoder/nli-deberta-v3-small"
print("Loading model...")
pipe = pipeline("text-classification", model=model_name)
res = pipe({"text": "A man is eating pizza", "text_pair": "A man is eating something"})
print(res)
