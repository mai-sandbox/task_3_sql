#!/usr/bin/env python3
"""
Simple test script to verify agent import and basic functionality
"""

try:
    print("Testing agent import...")
    import agent
    print("✓ Agent imported successfully")
    
    print("Testing app compilation...")
    app = agent.app
    print("✓ App compiled successfully")
    
    print("Testing basic state structure...")
    from langchain_core.messages import HumanMessage
    
    test_state = {
        "messages": [HumanMessage("Test message")]
    }
    print("✓ Basic state structure created")
    
    print("\nAll basic tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
