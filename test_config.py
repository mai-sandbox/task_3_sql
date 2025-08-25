#!/usr/bin/env python3
"""
Test script to verify LangGraph configuration file is properly structured
"""

import json
import os
from pathlib import Path

def test_langgraph_config():
    """Test the LangGraph configuration file"""
    
    print("=== Testing LangGraph Configuration ===\n")
    
    try:
        # Test 1: Configuration file exists
        config_path = Path("langgraph.json")
        if not config_path.exists():
            print("✗ langgraph.json file not found")
            return False
        print("✓ langgraph.json file exists")
        
        # Test 2: Configuration file is valid JSON
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✓ Configuration file is valid JSON")
        
        # Test 3: Required keys are present
        required_keys = ["dependencies", "graphs", "env"]
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing required key: {key}")
                return False
        print("✓ All required keys present")
        
        # Test 4: Dependencies are properly specified
        dependencies = config["dependencies"]
        if not isinstance(dependencies, list) or len(dependencies) == 0:
            print("✗ Dependencies must be a non-empty list")
            return False
        print(f"✓ Dependencies properly specified: {dependencies}")
        
        # Test 5: Graphs are properly configured
        graphs = config["graphs"]
        if not isinstance(graphs, dict) or len(graphs) == 0:
            print("✗ Graphs must be a non-empty dictionary")
            return False
        
        # Check agent graph specifically
        if "agent" not in graphs:
            print("✗ 'agent' graph not found in configuration")
            return False
        
        agent_path = graphs["agent"]
        if not agent_path.startswith("./agent.py:"):
            print("✗ Agent path should reference ./agent.py")
            return False
        print(f"✓ Agent graph properly configured: {agent_path}")
        
        # Test 6: Environment file is specified
        env_file = config["env"]
        if not isinstance(env_file, str):
            print("✗ Environment file must be a string")
            return False
        print(f"✓ Environment file specified: {env_file}")
        
        # Test 7: Agent file exists and exports app
        try:
            import agent
            if not hasattr(agent, 'app'):
                print("✗ agent.py does not export 'app' variable")
                return False
            print("✓ Agent file exists and exports 'app'")
        except ImportError as e:
            print(f"✗ Cannot import agent: {e}")
            return False
        
        # Test 8: Environment example file exists
        env_example_path = Path(".env.example")
        if env_example_path.exists():
            print("✓ .env.example file exists for reference")
        else:
            print("⚠ .env.example file not found (recommended for deployment)")
        
        print("\n=== Configuration Test Results ===")
        print("✓ LangGraph configuration is properly structured")
        print("✓ Ready for deployment")
        
        # Display configuration summary
        print(f"\nConfiguration Summary:")
        print(f"- Dependencies: {config['dependencies']}")
        print(f"- Graphs: {list(config['graphs'].keys())}")
        print(f"- Environment file: {config['env']}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in configuration file: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing configuration: {e}")
        return False

if __name__ == "__main__":
    success = test_langgraph_config()
    exit(0 if success else 1)
