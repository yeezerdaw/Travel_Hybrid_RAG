# test_hybrid_system.py
"""
Comprehensive test suite for Hybrid AI Travel Assistant
Run: python test_hybrid_system.py
"""

import asyncio
import time
from typing import Dict, List
import json

# Import your main module
try:
    from advanced_hybrid_chat import (
        embed_text,
        extract_query_intent,
        expand_query,
        rank_and_filter_results,
        process_query_async,
        process_query_sync
    )
    ASYNC_AVAILABLE = True
except ImportError:
    from hybrid_chat import (
        embed_text,
        pinecone_query,
        fetch_graph_context
    )
    ASYNC_AVAILABLE = False

# -----------------------------
# Test Cases
# -----------------------------
TEST_QUERIES = [
    # Simple queries
    "Tell me about Hanoi",
    "What are the best hotels in Hoi An?",
    "Show me attractions in Da Nang",
    
    # Complex queries
    "Create a romantic 4-day itinerary for Vietnam",
    "Find luxury hotels near beaches in central Vietnam",
    "What cultural attractions are connected to Hanoi?",
    
    # Edge cases
    "",  # Empty query
    "asdfghjkl",  # Nonsense
    "Hotels in Mars",  # Non-existent location
]

# -----------------------------
# Unit Tests
# -----------------------------
def test_embedding_generation():
    """Test 1: Embedding generation"""
    print("\n" + "="*70)
    print("TEST 1: Embedding Generation")
    print("="*70)
    
    test_text = "Hanoi is the capital of Vietnam"
    
    try:
        start = time.time()
        embedding = embed_text(test_text)
        elapsed = time.time() - start
        
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) > 0, "Embedding should not be empty"
        assert all(isinstance(x, float) for x in embedding), "All elements should be floats"
        
        print(f"✅ PASSED")
        print(f"   - Embedding dimension: {len(embedding)}")
        print(f"   - Time taken: {elapsed:.3f}s")
        print(f"   - Sample values: {embedding[:5]}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_caching():
    """Test 2: Embedding caching"""
    print("\n" + "="*70)
    print("TEST 2: Embedding Caching")
    print("="*70)
    
    test_text = "Test caching with this text"
    
    try:
        # First call (cache miss)
        start1 = time.time()
        emb1 = embed_text(test_text)
        time1 = time.time() - start1
        
        # Second call (cache hit)
        start2 = time.time()
        emb2 = embed_text(test_text)
        time2 = time.time() - start2
        
        assert emb1 == emb2, "Cached embedding should match original"
        
        speedup = time1 / time2 if time2 > 0 else float('inf')
        
        print(f"✅ PASSED")
        print(f"   - First call: {time1:.3f}s")
        print(f"   - Second call (cached): {time2:.3f}s")
        print(f"   - Speedup: {speedup:.1f}x")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

async def test_intent_extraction():
    """Test 3: Intent extraction"""
    print("\n" + "="*70)
    print("TEST 3: Intent Extraction")
    print("="*70)
    
    test_query = "Find romantic hotels in Hanoi for a 3-day trip"
    
    try:
        intent = await extract_query_intent(test_query)
        
        assert isinstance(intent, dict), "Intent should be a dictionary"
        
        print(f"✅ PASSED")
        print(f"   - Query: {test_query}")
        print(f"   - Extracted intent: {json.dumps(intent, indent=6)}")
        
        # Validate expected fields
        expected_fields = ["city", "type", "tags", "temporal", "trip_type"]
        for field in expected_fields:
            if field in intent:
                print(f"   - ✓ {field}: {intent[field]}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

async def test_query_expansion():
    """Test 4: Query expansion"""
    print("\n" + "="*70)
    print("TEST 4: Query Expansion")
    print("="*70)
    
    test_query = "romantic hotels Vietnam"
    
    try:
        expanded = await expand_query(test_query)
        
        assert isinstance(expanded, list), "Expanded queries should be a list"
        assert len(expanded) >= 1, "Should have at least original query"
        assert test_query in expanded, "Should include original query"
        
        print(f"✅ PASSED")
        print(f"   - Original: {test_query}")
        print(f"   - Expanded queries:")
        for i, q in enumerate(expanded, 1):
            print(f"     {i}. {q}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_result_ranking():
    """Test 5: Result ranking algorithm"""
    print("\n" + "="*70)
    print("TEST 5: Result Ranking Algorithm")
    print("="*70)
    
    # Mock data
    mock_matches = [
        {"id": "hotel_1", "score": 0.9, "metadata": {"name": "Hotel A", "type": "Hotel"}},
        {"id": "hotel_2", "score": 0.8, "metadata": {"name": "Hotel B", "type": "Hotel"}},
        {"id": "attraction_1", "score": 0.85, "metadata": {"name": "Temple", "type": "Attraction"}},
    ]
    
    mock_facts = [
        {"source_id": "hotel_1", "rel": "NEAR", "target_id": "attraction_1"},
        {"source_id": "hotel_1", "rel": "LOCATED_IN", "target_id": "city_hanoi"},
        {"source_id": "hotel_2", "rel": "LOCATED_IN", "target_id": "city_hanoi"},
    ]
    
    try:
        ranked = rank_and_filter_results(mock_matches, mock_facts, "test query", max_results=3)
        
        assert isinstance(ranked, list), "Ranked results should be a list"
        assert len(ranked) <= 3, "Should respect max_results"
        
        print(f"✅ PASSED")
        print(f"   - Input matches: {len(mock_matches)}")
        print(f"   - Graph facts: {len(mock_facts)}")
        print(f"   - Ranked results:")
        
        for i, (node_id, scores) in enumerate(ranked, 1):
            print(f"     {i}. {node_id}")
            print(f"        Vector: {scores['vector_score']:.3f}")
            print(f"        Graph degree: {scores['graph_degree']}")
            print(f"        Combined: {scores['combined_score']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

# -----------------------------
# Integration Tests
# -----------------------------
async def test_end_to_end_async():
    """Test 6: Full async pipeline"""
    print("\n" + "="*70)
    print("TEST 6: End-to-End Async Pipeline")
    print("="*70)
    
    test_query = "What are the best attractions in Hanoi?"
    
    try:
        start = time.time()
        answer = await process_query_async(test_query)
        elapsed = time.time() - start
        
        assert isinstance(answer, str), "Answer should be a string"
        assert len(answer) > 0, "Answer should not be empty"
        assert "attraction" in answer.lower() or "[" in answer, "Should mention attractions or node IDs"
        
        print(f"✅ PASSED")
        print(f"   - Query: {test_query}")
        print(f"   - Time: {elapsed:.2f}s")
        print(f"   - Answer length: {len(answer)} characters")
        print(f"   - Answer preview: {answer[:200]}...")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_end_to_end_sync():
    """Test 7: Full sync pipeline"""
    print("\n" + "="*70)
    print("TEST 7: End-to-End Sync Pipeline")
    print("="*70)
    
    test_query = "Hotels in Hoi An"
    
    try:
        start = time.time()
        answer = process_query_sync(test_query)
        elapsed = time.time() - start
        
        assert isinstance(answer, str), "Answer should be a string"
        assert len(answer) > 0, "Answer should not be empty"
        
        print(f"✅ PASSED")
        print(f"   - Query: {test_query}")
        print(f"   - Time: {elapsed:.2f}s")
        print(f"   - Answer length: {len(answer)} characters")
        print(f"   - Answer preview: {answer[:200]}...")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

async def test_edge_cases():
    """Test 8: Edge cases and error handling"""
    print("\n" + "="*70)
    print("TEST 8: Edge Cases & Error Handling")
    print("="*70)
    
    edge_cases = [
        ("", "Empty query"),
        ("x", "Single character"),
        ("?" * 100, "Very long query"),
    ]
    
    results = []
    
    for query, description in edge_cases:
        try:
            if ASYNC_AVAILABLE:
                answer = await process_query_async(query)
            else:
                answer = process_query_sync(query)
            
            status = "✓ Handled gracefully"
            results.append((description, status, True))
            
        except Exception as e:
            status = f"✗ Exception: {str(e)[:50]}"
            results.append((description, status, False))
    
    all_passed = all(r[2] for r in results)
    
    if all_passed:
        print(f"✅ PASSED")
    else:
        print(f"⚠️  PARTIAL PASS")
    
    for desc, status, _ in results:
        print(f"   - {desc}: {status}")
    
    return all_passed

# -----------------------------
# Performance Benchmark
# -----------------------------
async def benchmark_performance():
    """Test 9: Performance benchmark"""
    print("\n" + "="*70)
    print("TEST 9: Performance Benchmark")
    print("="*70)
    
    queries = [
        "Hotels in Hanoi",
        "Attractions in Hoi An",
        "Activities in Da Nang",
    ]
    
    times = []
    
    for query in queries:
        try:
            start = time.time()
            if ASYNC_AVAILABLE:
                await process_query_async(query)
            else:
                process_query_sync(query)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"   - {query}: {elapsed:.2f}s")
        except Exception as e:
            print(f"   - {query}: FAILED ({e})")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n✅ BENCHMARK COMPLETE")
        print(f"   - Average query time: {avg_time:.2f}s")
        print(f"   - Fastest: {min(times):.2f}s")
        print(f"   - Slowest: {max(times):.2f}s")
        
        # Performance rating
        if avg_time < 2.0:
            rating = "Excellent (< 2s)"
        elif avg_time < 3.5:
            rating = "Good (< 3.5s)"
        elif avg_time < 5.0:
            rating = "Acceptable (< 5s)"
        else:
            rating = "Needs optimization (> 5s)"
        
        print(f"   - Performance rating: {rating}")
        return True
    else:
        print(f"❌ BENCHMARK FAILED")
        return False

# -----------------------------
# Test Runner
# -----------------------------
async def run_all_tests():
    """Run all tests and generate report"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " " * 15 + "HYBRID AI TRAVEL ASSISTANT TEST SUITE" + " " * 15 + "║")
    print("╚" + "="*68 + "╝")
    
    results = []
    
    # Unit tests
    results.append(("Embedding Generation", test_embedding_generation()))
    results.append(("Embedding Caching", test_caching()))
    
    if ASYNC_AVAILABLE:
        results.append(("Intent Extraction", await test_intent_extraction()))
        results.append(("Query Expansion", await test_query_expansion()))
    
    results.append(("Result Ranking", test_result_ranking()))
    
    # Integration tests
    if ASYNC_AVAILABLE:
        results.append(("E2E Async Pipeline", await test_end_to_end_async()))
    
    results.append(("E2E Sync Pipeline", test_end_to_end_sync()))
    
    if ASYNC_AVAILABLE:
        results.append(("Edge Cases", await test_edge_cases()))
        results.append(("Performance Benchmark", await benchmark_performance()))
    
    # Generate report
    print("\n" + "="*70)
    print("TEST SUMMARY REPORT")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} | {test_name}")
    
    print("="*70)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is production-ready.")
    elif passed >= total * 0.8:
        print("\n⚠️  Most tests passed. Review failures before deployment.")
    else:
        print("\n❌ Multiple test failures. Debug required.")
    
    print("="*70 + "\n")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if ASYNC_AVAILABLE:
        asyncio.run(run_all_tests())
    else:
        print("Running in sync mode (advanced features unavailable)")
        # Run basic tests only
        test_embedding_generation()
        test_caching()
        test_end_to_end_sync()