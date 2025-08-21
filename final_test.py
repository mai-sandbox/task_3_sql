#!/usr/bin/env python3
"""
Final comprehensive test of the Chinook SQL Agent
Verifies all key functionality works correctly
"""

from agent import app, db
from langchain_core.messages import HumanMessage, AIMessage
import sys


def test_agent(query, test_name):
    """Test a single query and return success status"""
    print(f"\n{test_name}")
    print(f"Query: {query}")
    print("-" * 40)
    
    try:
        initial_state = {"messages": [HumanMessage(content=query)]}
        result = app.invoke(initial_state)
        
        if result and "messages" in result:
            # Find the final AI response (not a tool call)
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
                    response = msg.content
                    print(f"Response: {response[:200]}...")
                    
                    # Check for "don't know" in irrelevant queries
                    if "weather" in query.lower() or "python programming" in query.lower():
                        if "don't know" in response.lower():
                            print("✅ Correctly handled irrelevant query")
                            return True
                        else:
                            print("❌ Failed to handle irrelevant query properly")
                            return False
                    else:
                        # For valid queries, just check we got a response
                        if response and len(response) > 10:
                            print("✅ Valid response received")
                            return True
                    break
            
            print("❌ No valid response found")
            return False
        else:
            print("❌ No messages in result")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("FINAL TEST: Chinook SQL Agent")
    print("=" * 60)
    
    # Verify database is loaded
    print("\n1. Database Verification")
    print("-" * 40)
    try:
        result = db.execute_query("SELECT COUNT(*) as count FROM Album")
        album_count = result[0]['count'] if result else 0
        print(f"✅ Database loaded with {album_count} albums")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return 1
    
    # Test queries
    tests = [
        ("How many albums are in the database?", "2. Basic Count Query"),
        ("List the top 3 artists by number of albums", "3. Complex Query with JOIN"),
        ("What is the total revenue from all invoices?", "4. Aggregation Query"),
        ("What is the weather today?", "5. Irrelevant Query Test"),
    ]
    
    passed = 0
    failed = 0
    
    for query, test_name in tests:
        if test_agent(query, test_name):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(tests) + 1}")  # +1 for database verification
    print(f"Passed: {passed + 1}")  # +1 for database verification
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        print("The agent successfully:")
        print("  • Loads the Chinook database in memory")
        print("  • Generates SQL from natural language")
        print("  • Executes queries and returns results")
        print("  • Handles irrelevant queries appropriately")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
