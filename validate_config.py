#!/usr/bin/env python3
"""
Validate the LangGraph configuration file
"""

import json
import os

def validate_langgraph_config():
    """Validate the langgraph.json configuration file"""
    try:
        # Check if file exists
        if not os.path.exists('langgraph.json'):
            print("✗ langgraph.json file not found")
            return False
        
        # Load and validate JSON
        with open('langgraph.json', 'r') as f:
            config = json.load(f)
        
        print("✓ JSON is valid")
        
        # Check required fields
        required_fields = ['dependencies', 'graphs']
        for field in required_fields:
            if field not in config:
                print(f"✗ Missing required field: {field}")
                return False
            print(f"✓ Found required field: {field}")
        
        # Check dependencies
        if not isinstance(config['dependencies'], list):
            print("✗ Dependencies should be a list")
            return False
        
        print(f"✓ Dependencies list contains {len(config['dependencies'])} items")
        
        # Check graphs configuration
        if not isinstance(config['graphs'], dict):
            print("✗ Graphs should be a dictionary")
            return False
        
        if 'agent' not in config['graphs']:
            print("✗ Missing 'agent' graph configuration")
            return False
        
        agent_path = config['graphs']['agent']
        if not agent_path.endswith(':app'):
            print("✗ Agent graph should point to ':app' export")
            return False
        
        print(f"✓ Agent graph points to: {agent_path}")
        
        # Check if agent.py exists
        agent_file = agent_path.split(':')[0]
        if not os.path.exists(agent_file):
            print(f"✗ Agent file not found: {agent_file}")
            return False
        
        print(f"✓ Agent file exists: {agent_file}")
        
        print("\n✓ LangGraph configuration is valid and ready for deployment")
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False

if __name__ == "__main__":
    validate_langgraph_config()
