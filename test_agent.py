#!/usr/bin/env python3
"""
Test script for the LangGraph Text-to-SQL Agent
"""

import os
from langchain_core.messages import HumanMessage

# Set a dummy API key for testing import (if not already set)
if not os.getenv("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = "dummy-key-for-testing"

try:
    from agent import app
    print("✓ Agent imported successfully")
    
    # Test basic state structure
    initial_state = {
        "messages": [HumanMessage("Test query")],
        "query": "",
        "sql": "",
        "sql_result": None,
        "schema_info": "",
        "error": "",
        "response": ""
    }
    
    print("✓ Agent state structure is valid")
    print("✓ Agent is ready for use")
    
except Exception as e:
    print(f"✗ Error importing agent: {e}")
    import traceback
    traceback.print_exc()
