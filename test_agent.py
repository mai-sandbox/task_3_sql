#!/usr/bin/env python3
"""Test script for the LangGraph text-to-SQL agent."""

from langchain_core.messages import HumanMessage

def test_agent():
    try:
        # Import the agent
        from agent import app
        print("✓ Agent imported successfully")
        
        # Test basic functionality with a simple query
        initial_state = {
            "messages": [HumanMessage("What tables are in the database?")]
        }
        
        print("✓ Testing agent with schema query...")
        result = app.invoke(initial_state)
        
        if result and "messages" in result:
            print("✓ Agent executed successfully")
            print(f"✓ Response received: {len(result['messages'])} messages")
            return True
        else:
            print("✗ Agent execution failed - no messages in result")
            return False
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_agent()
    exit(0 if success else 1)
