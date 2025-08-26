#!/usr/bin/env python3
"""
Validation script for LangGraph configuration file.
"""

import json
import sys
import os

def validate_langgraph_config():
    """Validate the langgraph.json configuration file."""
    try:
        # Check if langgraph.json exists
        if not os.path.exists('langgraph.json'):
            print("❌ langgraph.json file not found")
            return False
        
        # Load and parse JSON
        with open('langgraph.json', 'r') as f:
            config = json.load(f)
        
        print("✅ langgraph.json is valid JSON")
        
        # Validate required keys
        required_keys = ['graphs', 'dependencies', 'env']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required key: {key}")
                return False
            print(f"✅ Found required key: {key}")
        
        # Validate graphs configuration
        graphs = config['graphs']
        if not graphs:
            print("❌ No graphs defined")
            return False
        
        for graph_name, graph_path in graphs.items():
            print(f"✅ Graph '{graph_name}' -> {graph_path}")
            
            # Check if the graph path references agent.py:app
            if not graph_path.endswith(':app'):
                print(f"❌ Graph '{graph_name}' should export 'app' variable")
                return False
            
            # Check if the referenced file exists
            file_path = graph_path.split(':')[0]
            if file_path.startswith('./'):
                file_path = file_path[2:]
            
            if not os.path.exists(file_path):
                print(f"❌ Graph file not found: {file_path}")
                return False
            
            print(f"✅ Graph file exists: {file_path}")
        
        # Validate dependencies
        dependencies = config['dependencies']
        if not dependencies:
            print("❌ No dependencies specified")
            return False
        
        print(f"✅ Dependencies: {dependencies}")
        
        # Validate environment file
        env_file = config['env']
        if env_file.startswith('./'):
            env_file = env_file[2:]
        
        if os.path.exists(env_file):
            print(f"✅ Environment file exists: {env_file}")
        else:
            print(f"⚠️  Environment file not found: {env_file} (this is optional)")
        
        # Check for optional but recommended keys
        if 'python_version' in config:
            print(f"✅ Python version specified: {config['python_version']}")
        else:
            print("⚠️  Python version not specified (recommended)")
        
        print("\n🎉 Configuration validation successful!")
        print("The langgraph.json file is properly configured for deployment.")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_langgraph_config()
    sys.exit(0 if success else 1)
