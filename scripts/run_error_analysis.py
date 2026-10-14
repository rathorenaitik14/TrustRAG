"""
Error Analysis Runner (Stage 4B)
================================
Categorizes at least 50 failure cases into interpretable failure buckets:
  1. Numerical contradiction missed
  2. Negation missed
  3. Paraphrase not recognized
  4. Retrieval failure
  5. Multi-hop claims
  6. Other unclassified
Outputs a table and saves both HTML and JSON reports to results/.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.error_analysis import main

if __name__ == "__main__":
    main()
