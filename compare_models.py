#!/usr/bin/env python3
"""Model comparison script for AI Golf Coaches.

This script helps compare different embedding and LLM models
to find the best performing combination for golf instruction content.
"""

import asyncio
import time
from pathlib import Path

from ai_golf_coaches.config import get_settings
from ai_golf_coaches.rag import EMBEDDING_MODELS, LLM_MODELS, ask, setup_models


def compare_embedding_models() -> None:
    """Compare different embedding models with the same query and LLM."""
    query = "How should I fix my slice off the tee? I tend to hit it way right."
    coach = "riley"
    
    print("🧪 EMBEDDING MODEL COMPARISON")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Coach: {coach}")
    print(f"LLM: gpt-4o (consistent)")
    print()
    
    results = []
    
    for name, model_name in EMBEDDING_MODELS.items():
        print(f"\n🔍 Testing embedding: {name} ({model_name})")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Setup with this embedding model
            setup_models(
                llm_model="gpt-4o", 
                embedding_model=model_name
            )
            
            # Get response
            response = ask(query, coach=coach)
            
            elapsed = time.time() - start_time
            
            print(f"⏱️  Response time: {elapsed:.2f}s")
            print(f"📝 Response length: {len(response)} chars")
            print(f"📄 Preview: {response[:200]}...")
            
            results.append({
                "name": name,
                "model": model_name,
                "time": elapsed,
                "response_length": len(response),
                "response": response,
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "name": name,
                "model": model_name,
                "time": None,
                "response_length": 0,
                "response": f"ERROR: {e}",
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 EMBEDDING COMPARISON SUMMARY")
    print("=" * 60)
    
    for result in results:
        if result["time"]:
            print(f"{result['name']:15} | {result['time']:6.2f}s | {result['response_length']:4d} chars")
        else:
            print(f"{result['name']:15} | ERROR   |    0 chars")


def compare_llm_models() -> None:
    """Compare different LLM models with the same query and embedding."""
    query = "What's the key to consistent ball striking with irons?"
    coach = "riley"
    
    print("\n\n🤖 LLM MODEL COMPARISON")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Coach: {coach}")
    print(f"Embedding: BAAI/bge-m3 (consistent)")
    print()
    
    results = []
    
    for name, model_name in LLM_MODELS.items():
        print(f"\n🧠 Testing LLM: {name} ({model_name})")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Setup with this LLM model
            setup_models(
                llm_model=model_name,
                embedding_model="BAAI/bge-m3"
            )
            
            # Get response
            response = ask(query, coach=coach)
            
            elapsed = time.time() - start_time
            
            print(f"⏱️  Response time: {elapsed:.2f}s")
            print(f"📝 Response length: {len(response)} chars")
            print(f"📄 Preview: {response[:200]}...")
            
            results.append({
                "name": name,
                "model": model_name,
                "time": elapsed,
                "response_length": len(response),
                "response": response,
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "name": name,
                "model": model_name,
                "time": None,
                "response_length": 0,
                "response": f"ERROR: {e}",
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 LLM COMPARISON SUMMARY")
    print("=" * 60)
    
    for result in results:
        if result["time"]:
            print(f"{result['name']:15} | {result['time']:6.2f}s | {result['response_length']:4d} chars")
        else:
            print(f"{result['name']:15} | ERROR   |    0 chars")


def test_best_combination() -> None:
    """Test the best performing combination."""
    query = "I'm struggling with my putting. How can I improve my distance control?"
    coach = "riley"
    
    print("\n\n🏆 TESTING RECOMMENDED COMBINATION")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Coach: {coach}")
    print("Embedding: BAAI/bge-m3")
    print("LLM: gpt-4o")
    print()
    
    start_time = time.time()
    
    # Setup best combination
    setup_models(
        llm_model="gpt-4o",
        embedding_model="BAAI/bge-m3"
    )
    
    response = ask(query, coach=coach)
    elapsed = time.time() - start_time
    
    print(f"⏱️  Response time: {elapsed:.2f}s")
    print(f"📝 Response length: {len(response)} chars")
    print("\n📋 FULL RESPONSE:")
    print("-" * 40)
    print(response)


def main() -> None:
    """Run model comparisons."""
    # Check if index exists
    index_dir = Path("data/index/youtube")
    if not index_dir.exists():
        print("❌ Vector index not found. Build it first with:")
        print("   poetry run python -m ai_golf_coaches.cli build-index")
        return
    
    print("🚀 AI Golf Coaches Model Comparison")
    print("This will test different embedding and LLM models")
    print("to find the best performing combination.\n")
    
    # Run comparisons
    compare_embedding_models()
    compare_llm_models()
    test_best_combination()
    
    print("\n" + "=" * 60)
    print("✅ Model comparison complete!")
    print("\nRecommendations:")
    print("• Best embedding: BAAI/bge-m3 (state-of-the-art retrieval)")
    print("• Best LLM: gpt-4o (latest GPT-4 for quality)")
    print("• Alternative: gpt-4-turbo (good balance of quality/cost)")


if __name__ == "__main__":
    main()