#!/usr/bin/env python3
"""Quick test of session mode performance."""

import time
from ai_golf_coaches.rag import load_index, get_query_engine, ask_with_engine

def test_session_performance():
    """Test session mode with pre-loaded index."""
    print("🏌️ Testing session mode performance...")
    
    # Time index loading (one-time cost)
    start_load = time.time()
    print("Loading index into memory...")
    index = load_index(use_test=False)  # Use full index
    query_engine = get_query_engine(index, coach="all")
    load_time = time.time() - start_load
    print(f"✅ Index loaded in {load_time:.2f} seconds")
    
    # Test multiple queries (fast after loading)
    questions = [
        "How do I fix my slice?",
        "How do I improve my putting?",
        "What's the best way to practice chipping?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n🤔 Question {i}: {question}")
        start_query = time.time()
        
        response = ask_with_engine(question, query_engine, coach="all")
        
        query_time = time.time() - start_query
        print(f"⏱️ Responded in {query_time:.2f} seconds")
        print(f"📖 Response: {response[:200]}...")

if __name__ == "__main__":
    test_session_performance()