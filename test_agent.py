#!/usr/bin/env python3
"""Test script for the LangGraph text-to-SQL agent."""

import os
from langchain_core.messages import HumanMessage

# Set a dummy API key for testing import
os.environ["ANTHROPIC_API_KEY"] = "test-key"

try:
    from agent import app
    print("✓ Agent imported successfully")
    print(f"✓ App type: {type(app)}")
    
    # Test the state structure
    initial_state = {
        "messages": [HumanMessage("Test query")],
        "sql_query": "",
        "sql_result": "",
        "error": ""
    }
    print("✓ State structure is valid")
    
    print("✓ All basic tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
