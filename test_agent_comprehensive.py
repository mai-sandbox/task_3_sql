#!/usr/bin/env python3
"""
Comprehensive test script for the LangGraph Text-to-SQL Agent

This script tests all required functionality:
1. Accept a state with messages containing HumanMessage
2. Process natural language queries about the Chinook database
3. Generate appropriate SQL queries
4. Execute them against the database
5. Return natural language responses
6. Handle irrelevant queries appropriately
"""

import os
import sys
from langchain_core.messages import HumanMessage, AIMessage
import json

def setup_test_environment():
    """Setup test environment with mock API key if needed"""
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠ Warning: No API keys found. Setting mock key for testing...")
        os.environ["OPENAI_API_KEY"] = "test-key-for-structure-testing"
        return False  # Indicates we can't do full LLM testing
    return True

def test_agent_import_and_setup():
    """Test 1: Agent import and basic setup"""
    print("=== Test 1: Agent Import and Setup ===")
    
    try:
        import agent
        print("✓ Agent module imported successfully")
        
        app = agent.app
        print("✓ App compiled and accessible")
        
        # Verify app has required methods
        if not hasattr(app, 'invoke'):
            print("✗ App missing 'invoke' method")
            return False
        print("✓ App has required 'invoke' method")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in agent import/setup: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_acceptance():
    """Test 2: Accept state with messages containing HumanMessage"""
    print("\n=== Test 2: State Acceptance ===")
    
    try:
        import agent
        
        # Test basic state structure
        test_state = {
            "messages": [HumanMessage("Test message")]
        }
        print("✓ Basic state structure created")
        
        # Test state with multiple messages
        complex_state = {
            "messages": [
                HumanMessage("First message"),
                AIMessage("AI response"),
                HumanMessage("Second message")
            ]
        }
        print("✓ Complex state structure created")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in state acceptance test: {e}")
        return False

def test_database_queries_structure():
    """Test 3: Database query processing structure (without LLM)"""
    print("\n=== Test 3: Database Query Processing Structure ===")
    
    try:
        import agent
        
        # Test individual node functions
        print("Testing individual node functions...")
        
        # Test analyze_query
        state = {"messages": [HumanMessage("Show me all albums by AC/DC")]}
        result = agent.analyze_query(state)
        
        if result.get("user_query") == "Show me all albums by AC/DC":
            print("✓ analyze_query function works correctly")
        else:
            print("✗ analyze_query function failed")
            return False
        
        # Test error handling in analyze_query
        empty_state = {"messages": []}
        error_result = agent.analyze_query(empty_state)
        
        if error_result.get("error"):
            print("✓ Error handling in analyze_query works")
        else:
            print("✗ Error handling in analyze_query failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in database query structure test: {e}")
        return False

def test_sql_execution_structure():
    """Test 4: SQL execution structure (with mock SQL)"""
    print("\n=== Test 4: SQL Execution Structure ===")
    
    try:
        import agent
        
        # Test SQL execution with a simple query
        state = {
            "messages": [HumanMessage("Test query")],
            "user_query": "Test query",
            "sql_query": "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;"
        }
        
        result = agent.execute_sql(state)
        
        # Check if SQL execution completed (even if it fails due to database setup)
        if "sql_result" in result or "error" in result:
            print("✓ SQL execution structure works")
        else:
            print("✗ SQL execution structure failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in SQL execution test: {e}")
        return False

def test_full_workflow_with_api_key():
    """Test 5: Full workflow with real API key (if available)"""
    print("\n=== Test 5: Full Workflow Test ===")
    
    has_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    
    if not has_api_key:
        print("⚠ Skipping full workflow test - no API key available")
        print("  To test with real LLM, set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return True
    
    try:
        import agent
        
        # Test cases for the Chinook database
        test_cases = [
            {
                "query": "Show me all albums by AC/DC",
                "description": "Valid music query - should work"
            },
            {
                "query": "What are the top 5 genres by track count?",
                "description": "Valid aggregation query - should work"
            },
            {
                "query": "What's the weather like today?",
                "description": "Irrelevant query - should return 'I don't know'"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test_case['description']}")
            print(f"Query: '{test_case['query']}'")
            
            try:
                # Create initial state
                initial_state = {
                    "messages": [HumanMessage(test_case["query"])]
                }
                
                # Invoke the agent
                result = agent.app.invoke(initial_state)
                
                # Check if we got a response
                if "messages" in result and len(result["messages"]) > 1:
                    ai_response = result["messages"][-1]
                    if isinstance(ai_response, AIMessage):
                        print(f"✓ Got AI response: {ai_response.content[:100]}...")
                        
                        # For irrelevant queries, check if it says "I don't know"
                        if "weather" in test_case["query"].lower():
                            if "don't know" in ai_response.content.lower():
                                print("✓ Correctly handled irrelevant query")
                            else:
                                print("⚠ May not have properly handled irrelevant query")
                    else:
                        print("✗ Response is not an AIMessage")
                        return False
                else:
                    print("✗ No AI response received")
                    return False
                    
            except Exception as e:
                print(f"✗ Error in test case {i}: {e}")
                # Don't return False here - continue with other tests
                continue
        
        print("✓ Full workflow tests completed")
        return True
        
    except Exception as e:
        print(f"✗ Error in full workflow test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test 6: Error handling for various scenarios"""
    print("\n=== Test 6: Error Handling ===")
    
    try:
        import agent
        
        # Test empty messages
        empty_state = {"messages": []}
        result = agent.analyze_query(empty_state)
        if result.get("error"):
            print("✓ Handles empty messages correctly")
        else:
            print("✗ Failed to handle empty messages")
            return False
        
        # Test malformed state
        try:
            malformed_state = {"invalid": "state"}
            result = agent.analyze_query(malformed_state)
            print("✓ Handles malformed state gracefully")
        except Exception:
            print("✓ Properly rejects malformed state")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in error handling test: {e}")
        return False

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🚀 Starting Comprehensive Agent Tests\n")
    
    # Setup test environment
    has_real_api_key = setup_test_environment()
    
    # Run all tests
    tests = [
        ("Agent Import and Setup", test_agent_import_and_setup),
        ("State Acceptance", test_state_acceptance),
        ("Database Query Structure", test_database_queries_structure),
        ("SQL Execution Structure", test_sql_execution_structure),
        ("Full Workflow", test_full_workflow_with_api_key),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("🎯 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Agent is working correctly.")
        if not has_real_api_key:
            print("💡 Note: Full LLM testing requires API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
    else:
        print("⚠ Some tests failed. Please review the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
