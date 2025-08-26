"""
Core functionality test for the LangGraph Text-to-SQL Agent

This script tests the essential components without requiring an OpenAI API key.
"""

import os
import sys

def test_database_functionality():
    """Test database initialization and basic operations."""
    print("🔧 Testing Database Functionality")
    print("-" * 40)
    
    try:
        # Temporarily set a dummy API key to avoid initialization errors
        os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"
        
        # Import the database class directly
        from agent import ChinookDatabase
        
        # Initialize database
        db = ChinookDatabase()
        
        # Test database connection
        if db.connection:
            print("✅ Database connection established")
        else:
            print("❌ Database connection failed")
            return False
            
        # Test schema extraction
        if db.schema_info and len(db.schema_info) > 0:
            print(f"✅ Schema extracted for {len(db.schema_info)} tables")
            
            # Verify expected tables
            expected_tables = {'Album', 'Artist', 'Customer', 'Employee', 'Genre', 
                             'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 
                             'PlaylistTrack', 'Track'}
            actual_tables = set(db.schema_info.keys())
            
            if expected_tables.issubset(actual_tables):
                print("✅ All 11 expected tables found")
                print(f"   Tables: {', '.join(sorted(actual_tables))}")
            else:
                missing = expected_tables - actual_tables
                print(f"❌ Missing tables: {missing}")
                return False
        else:
            print("❌ Schema extraction failed")
            return False
            
        # Test schema description
        schema_desc = db.get_schema_description()
        if schema_desc and len(schema_desc) > 500:
            print("✅ Schema description generated")
            print(f"   Description length: {len(schema_desc)} characters")
        else:
            print("❌ Schema description generation failed")
            return False
            
        # Test basic queries
        test_queries = [
            ("SELECT COUNT(*) FROM Artist", "Artist count"),
            ("SELECT COUNT(*) FROM Album", "Album count"),
            ("SELECT COUNT(*) FROM Track", "Track count"),
            ("SELECT Name FROM Genre LIMIT 3", "Sample genres"),
        ]
        
        print("\n📊 Testing database queries:")
        for query, description in test_queries:
            try:
                result = db.execute_query(query)
                if result and 'rows' in result and result['rows']:
                    if 'COUNT(*)' in query:
                        count = result['rows'][0][0]
                        print(f"   ✅ {description}: {count}")
                    else:
                        items = [row[0] for row in result['rows']]
                        print(f"   ✅ {description}: {', '.join(items)}")
                else:
                    print(f"   ❌ {description}: No results")
                    return False
            except Exception as e:
                print(f"   ❌ {description}: {str(e)}")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        return False


def test_relevance_detection():
    """Test query relevance detection."""
    print("\n🎯 Testing Query Relevance Detection")
    print("-" * 40)
    
    try:
        # Set dummy API key
        os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"
        
        from agent import is_relevant_query
        
        # Test cases
        relevant_queries = [
            "How many artists are there?",
            "Show me all albums",
            "What genres are available?",
            "Top selling tracks",
            "List customers",
            "Count songs in database",
            "Find rock music",
            "Show invoice totals"
        ]
        
        irrelevant_queries = [
            "What is the weather?",
            "Tell me about politics",
            "How to cook pasta?",
            "What is machine learning?",
            "Hello, how are you?",
            "What's 2 + 2?",
            "Tell me a joke",
            "What time is it?"
        ]
        
        print("Testing relevant queries:")
        relevant_passed = 0
        for query in relevant_queries:
            is_relevant = is_relevant_query(query)
            status = "✅" if is_relevant else "❌"
            print(f"   {status} '{query}' -> {is_relevant}")
            if is_relevant:
                relevant_passed += 1
                
        print(f"\nRelevant queries detected: {relevant_passed}/{len(relevant_queries)}")
        
        print("\nTesting irrelevant queries:")
        irrelevant_passed = 0
        for query in irrelevant_queries:
            is_relevant = is_relevant_query(query)
            status = "✅" if not is_relevant else "❌"
            print(f"   {status} '{query}' -> {is_relevant}")
            if not is_relevant:
                irrelevant_passed += 1
                
        print(f"Irrelevant queries rejected: {irrelevant_passed}/{len(irrelevant_queries)}")
        
        # Calculate success rate
        total_passed = relevant_passed + irrelevant_passed
        total_queries = len(relevant_queries) + len(irrelevant_queries)
        success_rate = (total_passed / total_queries) * 100
        
        print(f"\nOverall success rate: {success_rate:.1f}%")
        
        return success_rate >= 75  # Require at least 75% success rate
        
    except Exception as e:
        print(f"❌ Relevance detection test failed: {str(e)}")
        return False


def test_agent_structure():
    """Test agent structure and compilation."""
    print("\n🚀 Testing Agent Structure")
    print("-" * 30)
    
    try:
        # Set dummy API key
        os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"
        
        # Test imports
        from agent import AgentState, workflow, app
        
        print("✅ Agent modules imported successfully")
        
        # Test state schema
        if hasattr(AgentState, '__annotations__'):
            annotations = AgentState.__annotations__
            required_fields = {'messages', 'sql_query', 'sql_result', 'error'}
            actual_fields = set(annotations.keys())
            
            if required_fields.issubset(actual_fields):
                print("✅ AgentState schema has all required fields")
                print(f"   Fields: {', '.join(actual_fields)}")
            else:
                missing = required_fields - actual_fields
                print(f"❌ Missing AgentState fields: {missing}")
                return False
        else:
            print("❌ AgentState schema not properly defined")
            return False
            
        # Test workflow nodes
        if hasattr(workflow, 'nodes'):
            expected_nodes = {'generate_sql', 'execute_sql', 'generate_response'}
            actual_nodes = set(workflow.nodes.keys())
            
            if expected_nodes.issubset(actual_nodes):
                print("✅ Workflow has all required nodes")
                print(f"   Nodes: {', '.join(actual_nodes)}")
            else:
                missing = expected_nodes - actual_nodes
                print(f"❌ Missing workflow nodes: {missing}")
                return False
        else:
            print("❌ Workflow nodes not accessible")
            return False
            
        # Test app compilation
        if app:
            print("✅ Agent graph compiled successfully")
        else:
            print("❌ Agent graph not compiled")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Agent structure test failed: {str(e)}")
        return False


def test_configuration_file():
    """Test LangGraph configuration file."""
    print("\n📋 Testing Configuration File")
    print("-" * 35)
    
    try:
        import json
        
        # Check if langgraph.json exists
        if not os.path.exists('langgraph.json'):
            print("❌ langgraph.json file not found")
            return False
            
        # Load and validate configuration
        with open('langgraph.json', 'r') as f:
            config = json.load(f)
            
        print("✅ Configuration file loaded successfully")
        
        # Check required fields
        required_fields = ['dependencies', 'graphs']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
            else:
                print(f"✅ Found required field: {field}")
                
        # Check graph entry point
        if 'agent' in config['graphs']:
            entry_point = config['graphs']['agent']
            if entry_point == './agent.py:app':
                print("✅ Correct agent entry point configured")
            else:
                print(f"❌ Incorrect entry point: {entry_point}")
                return False
        else:
            print("❌ Agent graph not configured")
            return False
            
        # Check dependencies
        if './agent.py' in config['dependencies']:
            print("✅ Agent dependency configured")
        else:
            print("❌ Agent dependency not configured")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False


def run_comprehensive_test():
    """Run all tests and provide comprehensive results."""
    print("🧪 LangGraph Text-to-SQL Agent - Comprehensive Test Suite")
    print("=" * 65)
    
    tests = [
        ("Database Functionality", test_database_functionality),
        ("Relevance Detection", test_relevance_detection),
        ("Agent Structure", test_agent_structure),
        ("Configuration File", test_configuration_file),
    ]
    
    passed = 0
    total = len(tests)
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
                results.append(f"✅ {test_name}: PASSED")
            else:
                results.append(f"❌ {test_name}: FAILED")
        except Exception as e:
            results.append(f"❌ {test_name}: ERROR - {str(e)}")
    
    # Print results summary
    print("\n" + "=" * 65)
    print("📊 TEST RESULTS SUMMARY")
    print("-" * 25)
    
    for result in results:
        print(result)
    
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("✨ The LangGraph Text-to-SQL Agent is working correctly!")
        print("\n📋 Agent Features Validated:")
        print("   ✅ Chinook database initialization and schema extraction")
        print("   ✅ Query relevance detection")
        print("   ✅ LangGraph workflow structure and compilation")
        print("   ✅ Configuration file setup")
        print("\n🚀 The agent is ready for deployment and use!")
        
        print("\n💡 Sample queries you can try (with proper OpenAI API key):")
        print("   - 'How many artists are in the database?'")
        print("   - 'What are the top 5 albums by sales?'")
        print("   - 'List all genres available'")
        print("   - 'Who are the top customers by purchases?'")
        
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
