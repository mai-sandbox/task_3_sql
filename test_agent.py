#!/usr/bin/env python3
"""
Test script for the Chinook Text-to-SQL Agent
"""

import os
from langchain_core.messages import HumanMessage

# Set OpenAI API key if available
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set. Using placeholder.")
    os.environ["OPENAI_API_KEY"] = "placeholder-key"

try:
    from agent import app
    print("✓ Agent imported successfully")
    print(f"✓ Agent type: {type(app)}")
    
    # Test basic functionality with a simple query
    initial_state = {
        "messages": [HumanMessage("How many artists are in the database?")],
        "database_initialized": False,
        "schema_info": None,
        "user_query": None,
        "generated_sql": None,
        "sql_results": None,
        "error_message": None,
        "final_response": None
    }
    
    print("✓ Initial state created")
    print("✓ Agent is ready for testing")
    
    # Note: We can't run the full agent without a valid OpenAI API key
    # But we can verify the structure is correct
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
