"""
Simple test script for the LangGraph Text-to-SQL Agent

This script tests the core functionality without requiring API keys.
"""

def test_database_initialization():
    """Test database initialization and schema extraction."""
    print("🔧 Testing Database Initialization")
    print("-" * 40)
    
    try:
        from agent import db
        
        # Test database connection
        if db.connection:
            print("✅ Database connection established")
            
            # Test schema extraction
            if db.schema_info:
                print(f"✅ Schema extracted for {len(db.schema_info)} tables")
                print("📊 Available tables:")
                for table_name in db.schema_info.keys():
                    print(f"   - {table_name}")
                    
                # Verify we have the expected 11 tables
                expected_tables = {'Album', 'Artist', 'Customer', 'Employee', 'Genre', 
                                 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 
                                 'PlaylistTrack', 'Track'}
                actual_tables = set(db.schema_info.keys())
                
                if expected_tables.issubset(actual_tables):
                    print("✅ All expected tables found")
                else:
                    missing = expected_tables - actual_tables
                    print(f"❌ Missing tables: {missing}")
                    
            else:
                print("❌ Schema information not available")
                return False
                
            # Test schema description generation
            schema_desc = db.get_schema_description()
            if schema_desc and len(schema_desc) > 100:
                print("✅ Schema description generated successfully")
                print(f"   Description length: {len(schema_desc)} characters")
            else:
                print("❌ Schema description generation failed")
                return False
                
        else:
            print("❌ Database connection not established")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Database initialization test failed: {str(e)}")
        return False


def test_database_queries():
    """Test basic database queries."""
    print("\n🔍 Testing Database Queries")
    print("-" * 30)
    
    try:
        from agent import db
        
        # Test queries
        test_queries = [
            ("SELECT COUNT(*) FROM Artist", "Artist count"),
            ("SELECT COUNT(*) FROM Album", "Album count"),
            ("SELECT COUNT(*) FROM Track", "Track count"),
            ("SELECT COUNT(*) FROM Customer", "Customer count"),
            ("SELECT Name FROM Genre LIMIT 5", "First 5 genres"),
        ]
        
        for query, description in test_queries:
            try:
                result = db.execute_query(query)
                if result and 'rows' in result and result['rows']:
                    if 'COUNT(*)' in query:
                        count = result['rows'][0][0]
                        print(f"✅ {description}: {count}")
                    else:
                        items = [row[0] for row in result['rows']]
                        print(f"✅ {description}: {', '.join(items)}")
                else:
                    print(f"❌ {description}: No results")
                    
            except Exception as e:
                print(f"❌ {description}: Query failed - {str(e)}")
                
        return True
        
    except Exception as e:
        print(f"❌ Database query test failed: {str(e)}")
        return False


def test_relevance_detection():
    """Test query relevance detection."""
    print("\n🎯 Testing Query Relevance Detection")
    print("-" * 40)
    
    try:
        from agent import is_relevant_query
        
        # Test relevant queries
        relevant_queries = [
            "How many artists are there?",
            "Show me all albums",
            "What genres are available?",
            "Top selling tracks",
            "List all customers",
            "How many songs are in the database?"
        ]
        
        # Test irrelevant queries
        irrelevant_queries = [
            "What is the weather?",
            "Tell me about politics",
            "How to cook pasta?",
            "What is machine learning?",
            "Hello, how are you?",
            "What's 2 + 2?"
        ]
        
        print("Testing relevant queries:")
        relevant_passed = 0
        for query in relevant_queries:
            is_relevant = is_relevant_query(query)
            status = "✅" if is_relevant else "❌"
            print(f"   {status} '{query}' -> {is_relevant}")
            if is_relevant:
                relevant_passed += 1
                
        print(f"\nRelevant queries passed: {relevant_passed}/{len(relevant_queries)}")
        
        print("\nTesting irrelevant queries:")
        irrelevant_passed = 0
        for query in irrelevant_queries:
            is_relevant = is_relevant_query(query)
            status = "✅" if not is_relevant else "❌"
            print(f"   {status} '{query}' -> {is_relevant}")
            if not is_relevant:
                irrelevant_passed += 1
                
        print(f"Irrelevant queries passed: {irrelevant_passed}/{len(irrelevant_queries)}")
        
        # Overall success rate
        total_passed = relevant_passed + irrelevant_passed
        total_queries = len(relevant_queries) + len(irrelevant_queries)
        success_rate = (total_passed / total_queries) * 100
        
        print(f"\nOverall relevance detection success rate: {success_rate:.1f}%")
        
        return success_rate >= 70  # At least 70% success rate
        
    except Exception as e:
        print(f"❌ Relevance detection test failed: {str(e)}")
        return False


def test_agent_import():
    """Test that the agent can be imported and compiled."""
    print("\n🚀 Testing Agent Import and Compilation")
    print("-" * 45)
    
    try:
        from agent import app, workflow
        
        print("✅ Agent module imported successfully")
        
        # Check if app is compiled
        if app:
            print("✅ Agent graph compiled successfully")
            
            # Check if workflow has the expected nodes
            if hasattr(workflow, 'nodes'):
                expected_nodes = {'generate_sql', 'execute_sql', 'generate_response'}
                actual_nodes = set(workflow.nodes.keys())
                
                if expected_nodes.issubset(actual_nodes):
                    print("✅ All expected workflow nodes found")
                    print(f"   Nodes: {', '.join(actual_nodes)}")
                else:
                    missing = expected_nodes - actual_nodes
                    print(f"❌ Missing workflow nodes: {missing}")
                    return False
            else:
                print("⚠️  Cannot verify workflow nodes")
                
            return True
        else:
            print("❌ Agent graph not compiled")
            return False
            
    except Exception as e:
        print(f"❌ Agent import test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and provide summary."""
    print("🧪 LangGraph Text-to-SQL Agent Test Suite")
    print("=" * 50)
    
    tests = [
        ("Database Initialization", test_database_initialization),
        ("Database Queries", test_database_queries),
        ("Relevance Detection", test_relevance_detection),
        ("Agent Import", test_agent_import),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n❌ {test_name}: ERROR - {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Agent is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n📋 Sample queries you can try:")
        print("   - 'How many artists are in the database?'")
        print("   - 'What are the top 5 albums by sales?'")
        print("   - 'List all genres available'")
        print("   - 'Who are the top customers by purchases?'")
        print("\n💡 Note: Full agent testing requires an OpenAI API key")
        print("   Set OPENAI_API_KEY environment variable to test with LLM")
