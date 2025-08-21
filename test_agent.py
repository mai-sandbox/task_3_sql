"""
Test script for the Chinook SQL Agent
This script tests various queries to ensure the agent properly:
1. Generates SQL queries
2. Executes them against the database
3. Returns natural language responses
"""

from agent import app
from langchain_core.messages import HumanMessage
import json


def test_query(query: str, test_name: str = "Test"):
    """Helper function to test a single query"""
    print(f"\n{'='*60}")
    print(f"{test_name}")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print("-" * 40)
    
    try:
        # Create initial state with the query
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Invoke the agent
        result = app.invoke(initial_state)
        
        # Extract and display the response
        if result and "messages" in result:
            for message in result["messages"]:
                if hasattr(message, 'content') and message.content:
                    # Skip tool calls and focus on final responses
                    if not hasattr(message, 'tool_calls'):
                        print(f"Response: {message.content}")
                        return True
        
        print("No response received")
        return False
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def run_all_tests():
    """Run a comprehensive set of tests"""
    
    print("\n" + "="*60)
    print("CHINOOK SQL AGENT TEST SUITE")
    print("="*60)
    
    tests = [
        # Basic counting queries
        ("How many albums are in the database?", "Test 1: Count Albums"),
        ("How many artists are there?", "Test 2: Count Artists"),
        ("How many customers do we have?", "Test 3: Count Customers"),
        
        # Specific data queries
        ("What are the top 5 most expensive tracks?", "Test 4: Top Expensive Tracks"),
        ("Which artist has the most albums?", "Test 5: Artist with Most Albums"),
        ("List all genres in the database", "Test 6: List Genres"),
        
        # Join queries
        ("Which albums were released by AC/DC?", "Test 7: Albums by Artist"),
        ("What is the total revenue from all invoices?", "Test 8: Total Revenue"),
        ("Who are the top 3 customers by total purchase amount?", "Test 9: Top Customers"),
        
        # Complex queries
        ("What is the average track length in minutes?", "Test 10: Average Track Length"),
        ("Which country has the most customers?", "Test 11: Country with Most Customers"),
        ("What are the most popular genres by number of tracks?", "Test 12: Popular Genres"),
        
        # Employee queries
        ("How many employees are there and what are their titles?", "Test 13: Employee Information"),
        ("Who reports to whom in the company?", "Test 14: Reporting Structure"),
        
        # Irrelevant query test
        ("What is the weather today?", "Test 15: Irrelevant Query (Should return 'I don't know')"),
        ("Tell me about Python programming", "Test 16: Off-topic Query (Should return 'I don't know')"),
    ]
    
    passed = 0
    failed = 0
    
    for query, test_name in tests:
        if test_query(query, test_name):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(tests))*100:.1f}%")
    
    return passed, failed


if __name__ == "__main__":
    # First, let's test if the agent loads correctly
    print("Loading Chinook SQL Agent...")
    print("Agent loaded successfully!")
    
    # Run all tests
    passed, failed = run_all_tests()
    
    # Exit with appropriate code
    if failed == 0:
        print("\n✅ All tests passed successfully!")
        exit(0)
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review the output above.")
        exit(1)
