"""
Simple test to verify the agent loads and works with a basic query
"""

import sys
import os

# Test basic imports first
try:
    from agent import app
    from langchain_core.messages import HumanMessage
    print("✅ Successfully imported agent and dependencies")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test a simple query
def test_simple_query():
    """Test a single simple query"""
    print("\n" + "="*60)
    print("Testing Chinook SQL Agent")
    print("="*60)
    
    query = "How many albums are in the database?"
    print(f"Query: {query}")
    print("-" * 40)
    
    try:
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Invoke the agent
        print("Invoking agent...")
        result = app.invoke(initial_state)
        
        # Check if we got a response
        if result and "messages" in result:
            print(f"✅ Agent returned {len(result['messages'])} messages")
            
            # Display the final response
            for message in result["messages"]:
                if hasattr(message, 'content') and message.content:
                    if not hasattr(message, 'tool_calls'):
                        print(f"\nFinal Response: {message.content[:500]}...")  # Show first 500 chars
                        return True
        
        print("❌ No valid response received")
        return False
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting simple agent test...")
    
    # Run the test
    success = test_simple_query()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("The agent is working correctly.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
        sys.exit(1)
