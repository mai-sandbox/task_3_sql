#!/usr/bin/env python3
"""
Test script to verify the LangGraph text-to-SQL agent is properly implemented.
"""

import sys
import os

def test_agent_structure():
    """Test that the agent module has the required structure."""
    try:
        import agent
        print("✅ Agent module imported successfully")
        
        # Check if app variable exists
        if hasattr(agent, 'app'):
            print("✅ App variable exists and is exported")
        else:
            print("❌ App variable not found")
            return False
            
        # Check if required tools exist
        required_tools = [
            'initialize_database',
            'validate_query_relevance', 
            'generate_sql_query',
            'execute_sql_query',
            'format_natural_language_response'
        ]
        
        for tool_name in required_tools:
            if hasattr(agent, tool_name):
                print(f"✅ Tool '{tool_name}' exists")
            else:
                print(f"❌ Tool '{tool_name}' not found")
                return False
        
        # Check if create_react_agent is used
        import inspect
        source = inspect.getsource(agent.create_sql_agent)
        if 'create_react_agent' in source:
            print("✅ Uses create_react_agent from langgraph.prebuilt")
        else:
            print("❌ Does not use create_react_agent")
            return False
            
        print("✅ All structural requirements met")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import agent module: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing agent structure: {e}")
        return False

def test_database_url():
    """Test that the correct Chinook database URL is used."""
    try:
        import agent
        import inspect
        
        source = inspect.getsource(agent.initialize_database)
        expected_url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
        
        if expected_url in source:
            print("✅ Correct Chinook database URL is used")
            return True
        else:
            print("❌ Incorrect or missing Chinook database URL")
            return False
            
    except Exception as e:
        print(f"❌ Error testing database URL: {e}")
        return False

def test_schema_context():
    """Test that schema context is included."""
    try:
        import agent
        
        if hasattr(agent, 'CHINOOK_SCHEMA') and agent.CHINOOK_SCHEMA:
            print("✅ Database schema context is defined")
            
            # Check for key tables
            key_tables = ['Artist', 'Album', 'Track', 'Customer', 'Invoice']
            schema_text = agent.CHINOOK_SCHEMA
            
            for table in key_tables:
                if table in schema_text:
                    print(f"✅ Schema includes {table} table")
                else:
                    print(f"❌ Schema missing {table} table")
                    return False
            return True
        else:
            print("❌ Database schema context not found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing schema context: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing LangGraph Text-to-SQL Agent Implementation")
    print("=" * 50)
    
    tests = [
        test_agent_structure,
        test_database_url,
        test_schema_context
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The agent is properly implemented.")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
