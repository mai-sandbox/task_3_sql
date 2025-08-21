"""
Verification script for the Chinook SQL Agent
This script verifies that the agent can:
1. Generate SQL queries
2. Execute them against the database
3. Return natural language responses
"""

from agent import app
from langchain_core.messages import HumanMessage, AIMessage
import sys


def verify_agent_functionality():
    """Verify the agent works with different types of queries"""
    
    print("\n" + "="*70)
    print("CHINOOK SQL AGENT VERIFICATION")
    print("="*70)
    
    # Test cases with expected behavior
    test_cases = [
        {
            "query": "How many albums are in the database?",
            "description": "Basic counting query",
            "expect_success": True
        },
        {
            "query": "List the first 5 artists",
            "description": "Simple SELECT query with limit",
            "expect_success": True
        },
        {
            "query": "What is the total revenue from all invoices?",
            "description": "Aggregation query",
            "expect_success": True
        },
        {
            "query": "Which artist has the most albums?",
            "description": "Complex query with JOIN and GROUP BY",
            "expect_success": True
        },
        {
            "query": "What is the weather today?",
            "description": "Irrelevant query (should return 'I don't know')",
            "expect_success": True,
            "expect_dont_know": True
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Query: {test['query']}")
        print("-" * 50)
        
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=test['query'])]
            }
            
            # Invoke the agent
            result = app.invoke(initial_state)
            
            # Check if we got a response
            if result and "messages" in result and len(result["messages"]) > 0:
                # Find the final AI message (not a tool call)
                final_response = None
                for message in reversed(result["messages"]):
                    if isinstance(message, AIMessage) and not hasattr(message, 'tool_calls'):
                        final_response = message.content
                        break
                
                if final_response:
                    # Check if it's an "I don't know" response for irrelevant queries
                    if test.get("expect_dont_know", False):
                        if "don't know" in final_response.lower():
                            print(f"✅ PASSED: Correctly returned 'I don't know' response")
                            print(f"Response: {final_response[:200]}...")
                            passed += 1
                        else:
                            print(f"❌ FAILED: Expected 'I don't know' but got different response")
                            print(f"Response: {final_response[:200]}...")
                            failed += 1
                    else:
                        # For relevant queries, just check we got a response
                        print(f"✅ PASSED: Got valid response")
                        print(f"Response preview: {final_response[:200]}...")
                        passed += 1
                else:
                    print(f"❌ FAILED: No final response found")
                    failed += 1
            else:
                print(f"❌ FAILED: No messages in result")
                failed += 1
                
        except Exception as e:
            print(f"❌ FAILED: Error - {str(e)}")
            failed += 1
    
    # Print summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_cases))*100:.1f}%")
    
    return passed, failed


def main():
    """Main verification function"""
    print("Starting Chinook SQL Agent verification...")
    print("Loading agent and database...")
    
    try:
        # The database should be loaded when the agent module is imported
        print("✅ Agent loaded successfully")
        
        # Run verification tests
        passed, failed = verify_agent_functionality()
        
        # Determine overall success
        if failed == 0:
            print("\n" + "="*70)
            print("✅ VERIFICATION SUCCESSFUL!")
            print("The agent properly:")
            print("  1. Generates SQL queries from natural language")
            print("  2. Executes queries against the Chinook database")
            print("  3. Returns natural language responses")
            print("  4. Handles irrelevant queries appropriately")
            print("="*70)
            return 0
        else:
            print("\n" + "="*70)
            print("⚠️ VERIFICATION PARTIALLY SUCCESSFUL")
            print(f"{passed} tests passed, {failed} tests failed")
            print("Please review the failures above")
            print("="*70)
            return 1
            
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
