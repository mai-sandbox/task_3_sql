"""
Test script for the LangGraph Text-to-SQL Agent

This script tests the core functionality without requiring API keys.
"""

import sqlite3
import requests
from agent import setup_database, generate_sql_query, execute_sql_query

def test_database_setup():
    """Test the database setup functionality."""
    print("Testing database setup...")
    result = setup_database.invoke({})
    print(f"Database setup result: {result}")
    return "successful" in result.lower()

def test_sql_generation():
    """Test SQL query generation for various types of queries."""
    print("\nTesting SQL query generation...")
    
    test_queries = [
        "How many artists are in the database?",
        "Show me all albums",
        "What are the longest tracks?",
        "List all genres",
        "Who are the top customers?",
        "What's the weather like today?",  # Irrelevant query
        "Tell me about politics"  # Irrelevant query
    ]
    
    results = []
    for query in test_queries:
        print(f"\nQuery: {query}")
        sql_result = generate_sql_query.invoke({"natural_language_query": query})
        print(f"Generated SQL: {sql_result}")
        results.append((query, sql_result))
    
    return results

def test_sql_execution():
    """Test SQL query execution."""
    print("\nTesting SQL query execution...")
    
    # Test queries that should work
    test_sqls = [
        "SELECT COUNT(*) as artist_count FROM Artist;",
        "SELECT Name FROM Genre ORDER BY Name LIMIT 5;",
        "SELECT COUNT(*) as album_count FROM Album;",
        "I don't know - this query doesn't appear to be related to the music database."
    ]
    
    results = []
    for sql in test_sqls:
        print(f"\nExecuting: {sql}")
        result = execute_sql_query.invoke({"sql_query": sql})
        print(f"Result: {result}")
        results.append((sql, result))
    
    return results

def test_full_workflow():
    """Test the complete workflow: setup -> generate -> execute."""
    print("\n" + "="*60)
    print("TESTING COMPLETE WORKFLOW")
    print("="*60)
    
    # Step 1: Setup database
    print("\n1. Setting up database...")
    setup_result = setup_database.invoke({})
    print(f"Setup result: {setup_result}")
    
    if "successful" not in setup_result.lower():
        print("Database setup failed, cannot continue workflow test")
        return False
    
    # Step 2: Test various queries
    test_cases = [
        {
            "query": "How many artists are in the database?",
            "expected_type": "count"
        },
        {
            "query": "Show me some albums with their artists",
            "expected_type": "list"
        },
        {
            "query": "What genres are available?",
            "expected_type": "list"
        },
        {
            "query": "What's the capital of France?",
            "expected_type": "irrelevant"
        }
    ]
    
    workflow_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing query: {test_case['query']}")
        
        # Generate SQL
        sql_query = generate_sql_query.invoke({"natural_language_query": test_case['query']})
        print(f"   Generated SQL: {sql_query}")
        
        # Execute SQL
        execution_result = execute_sql_query.invoke({"sql_query": sql_query})
        print(f"   Execution result: {execution_result}")
        
        workflow_results.append({
            "query": test_case['query'],
            "sql": sql_query,
            "result": execution_result,
            "expected_type": test_case['expected_type']
        })
    
    return workflow_results

def analyze_results(workflow_results):
    """Analyze the test results and provide a summary."""
    print("\n" + "="*60)
    print("TEST RESULTS ANALYSIS")
    print("="*60)
    
    total_tests = len(workflow_results)
    successful_tests = 0
    
    for result in workflow_results:
        query = result['query']
        sql = result['sql']
        execution_result = result['result']
        expected_type = result['expected_type']
        
        print(f"\nQuery: {query}")
        print(f"Expected: {expected_type}")
        
        # Check if the result matches expectations
        if expected_type == "irrelevant":
            if "I don't know" in sql or "I don't know" in execution_result:
                print("✅ PASS: Correctly identified irrelevant query")
                successful_tests += 1
            else:
                print("❌ FAIL: Should have responded 'I don't know'")
        
        elif expected_type == "count":
            if "COUNT" in sql.upper() and ("error" not in execution_result.lower()):
                print("✅ PASS: Generated count query and executed successfully")
                successful_tests += 1
            else:
                print("❌ FAIL: Count query failed")
        
        elif expected_type == "list":
            if "SELECT" in sql.upper() and ("error" not in execution_result.lower()):
                print("✅ PASS: Generated list query and executed successfully")
                successful_tests += 1
            else:
                print("❌ FAIL: List query failed")
    
    print(f"\n" + "="*60)
    print(f"OVERALL RESULTS: {successful_tests}/{total_tests} tests passed")
    print("="*60)
    
    return successful_tests == total_tests

def main():
    """Run all tests."""
    print("LangGraph Text-to-SQL Agent - Core Functionality Tests")
    print("="*60)
    
    try:
        # Test individual components
        db_setup_success = test_database_setup()
        sql_generation_results = test_sql_generation()
        sql_execution_results = test_sql_execution()
        
        # Test complete workflow
        workflow_results = test_full_workflow()
        
        # Analyze results
        all_tests_passed = analyze_results(workflow_results)
        
        if all_tests_passed:
            print("\n🎉 ALL TESTS PASSED! The agent core functionality is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Please review the results above.")
            
        print("\nNOTE: To test the full LangGraph agent with LLM integration:")
        print("1. Set up your Anthropic API key: export ANTHROPIC_API_KEY='your-key-here'")
        print("2. Run: python agent.py")
        print("3. Or use the agent in your own script by importing the 'app' variable")
        
    except Exception as e:
        print(f"Test execution failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
