import os
import sys
import json
from dotenv import load_dotenv

from claims.claim_extractor import ClaimExtractor
from generation.llm_client import get_llm_client

# Force UTF-8 for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    load_dotenv()
    
    if not os.getenv("LLM_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        print("ERROR: LLM_API_KEY or GEMINI_API_KEY is not set.")
        sys.exit(1)
        
    print("Initializing Claim Extractor...")
    client = get_llm_client()
    extractor = ClaimExtractor(client=client)
    
    # Pre-generated answers to bypass earlier pipeline steps and avoid 429 quota exhaustion.
    test_cases = [
        {
            "question": "What is the goal of data mining?",
            "answer": "Based on the provided evidence, the goal of data mining is to provide efficient tools and techniques for Knowledge Discovery in Databases (KDD)."
        },
        {
            "question": "What is data preprocessing?",
            "answer": "Data preprocessing is a process of transforming raw data into an understandable format. Its four major tasks are data cleaning, data integration, data reduction, and data transformation."
        },
        {
            "question": "What is KDD?",
            "answer": "KDD stands for Knowledge Discovery in Databases. It is the overall process of discovering useful knowledge from data."
        },
        {
            "question": "What are measures of central tendency?",
            "answer": "Mean, median, and mode are the three measures of central tendency. The mean involves all the values in the dataset."
        },
        {
            "question": "CONTROLLED FALSE-CLAIM TEST",
            "answer": "The goal of data mining is to manufacture physical computers."
        }
    ]
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n=======================================================")
        print(f"TEST CASE {i}")
        print(f"=======================================================\n")
        
        q = tc["question"]
        a = tc["answer"]
        
        print(f"QUESTION:\n{q}\n")
        print(f"GENERATED ANSWER:\n{a}\n")
        
        print("EXTRACTING CLAIMS...")
        
        try:
            # We use a backoff retry just in case the API limit hits even here.
            import time
            max_retries = 6
            result = None
            for attempt in range(max_retries):
                try:
                    result = extractor.extract_claims(a, question=q)
                    break
                except Exception as e:
                    if ("429" in str(e) or "503" in str(e)) and attempt < max_retries - 1:
                        print(f"Rate limited. Retrying in {10 + (2**attempt)} seconds...")
                        time.sleep(10 + (2 ** attempt))
                        continue
                    raise e
                    
            print("\nEXTRACTED CLAIMS:\n")
            for claim in result["claims"]:
                print(f"{claim['claim_id']}:\n{claim['claim_text']} (Type: {claim['claim_type']})\n")
                
        except Exception as e:
            print(f"Extraction failed: {e}")

if __name__ == "__main__":
    main()
