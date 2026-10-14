"""
conftest.py — pytest configuration for TrustRAG
Adds the project root to sys.path so all modules are importable.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
