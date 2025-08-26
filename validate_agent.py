#!/usr/bin/env python3
"""Validate that the agent follows the required minimal state pattern"""

from langchain_core.messages import HumanMessage
from agent import app

def validate_agent():
    """Test that agent accepts minimal state with only messages"""
    
    print("Validating agent requirements...")
    print("-" * 40)
    
    # Test 1: Minimal state with only messages
    initial_state = {
        "messages": [HumanMessage("How many artists are in the database?")]
    }
    
    try:
        result = app.invoke(initial_state)
        print("✅ Agent accepts minimal state with messages")
        print(f"   Response: {result['messages'][-1].content[:100]}...")
    except Exception as e:
        print(f"❌ Failed to invoke with minimal state: {e}")
        return False
    
    # Test 2: Check export pattern
    try:
        from agent import app as imported_app
        assert imported_app is not None
        print("✅ Agent exports compiled graph as 'app'")
    except Exception as e:
        print(f"❌ Failed to import app: {e}")
        return False
    
    # Test 3: Test with edge case query
    edge_state = {
        "messages": [HumanMessage("Tell me about quantum physics")]
    }
    
    try:
        result = app.invoke(edge_state)
        response = result['messages'][-1].content.lower()
        if "don't know" in response or "cannot answer" in response or "chinook" in response:
            print("✅ Agent correctly handles out-of-scope queries")
        else:
            print("⚠️  Agent may not be handling out-of-scope queries correctly")
    except Exception as e:
        print(f"❌ Failed on edge case: {e}")
        return False
    
    print("-" * 40)
    print("✅ All validation checks passed!")
    return True

if __name__ == "__main__":
    validate_agent()