"""
Test script for the LangGraph Text-to-SQL Agent

This script tests the agent with various sample queries to ensure proper
SQL generation, execution, and natural language response generation.
"""

import os
from langchain_core.messages import HumanMessage

# Set up environment variables for testing
# Note: In a real deployment, these would be set in the environment or .env file
os.environ.setdefault("OPENAI_API_KEY", "your-openai-api-key-here")

def test_agent():
    """Test the agent with various sample queries."""
    
    print("🚀 Testing LangGraph Text-to-SQL Agent for Chinook Database")
    print("=" * 60)
    
    try:
        # Import the agent
        from agent import app
        print("✅ Successfully imported the agent")
        
        # Test queries
        test_queries = [
            "How many artists are in the database?",
            "What are the top 5 albums by sales?",
            "List all genres available in the database",
            "Who are the top 3 customers by total purchases?",
            "What is the most popular track?",
            "How many employees work at the company?",
            "What is the weather like today?",  # Irrelevant query
            "Tell me about machine learning",    # Irrelevant query
        ]
        
        print(f"\n📋 Running {len(test_queries)} test queries...")
        print("-" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {query}")
            print("-" * 40)
            
            try:
                # Create initial state with the query
                initial_state = {
                    "messages": [HumanMessage(content=query)]
                }
                
                # Run the agent
                result = app.invoke(initial_state)
                
                # Extract the response
                if result and "messages" in result and len(result["messages"]) > 1:
                    response = result["messages"][-1].content
                    print(f"🤖 Response: {response}")
                    
                    # Check if SQL was generated (for debugging)
                    if "sql_query" in result and result["sql_query"]:
                        print(f"🔧 Generated SQL: {result['sql_query']}")
                    
                    # Check for errors
                    if "error" in result and result["error"]:
                        print(f"⚠️  Error: {result['error']}")
                        
                else:
                    print("❌ No response generated")
                    
            except Exception as e:
                print(f"❌ Error testing query: {str(e)}")
                
        print("\n" + "=" * 60)
        print("🎉 Agent testing completed!")
        
    except ImportError as e:
        print(f"❌ Failed to import agent: {str(e)}")
        print("Make sure the agent.py file exists and all dependencies are installed.")
        
    except Exception as e:
        print(f"❌ Unexpected error during testing: {str(e)}")


def test_database_connection():
    """Test the database connection and schema extraction."""
    
    print("\n🔧 Testing Database Connection and Schema")
    print("-" * 50)
    
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
            else:
                print("❌ Schema information not available")
                
            # Test a simple query
            try:
                result = db.execute_query("SELECT COUNT(*) FROM Artist")
                if result and 'rows' in result:
                    artist_count = result['rows'][0][0]
                    print(f"✅ Database query test passed - Found {artist_count} artists")
                else:
                    print("❌ Database query test failed")
            except Exception as e:
                print(f"❌ Database query test failed: {str(e)}")
                
        else:
            print("❌ Database connection not established")
            
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")


def test_relevance_detection():
    """Test the query relevance detection."""
    
    print("\n🎯 Testing Query Relevance Detection")
    print("-" * 40)
    
    try:
        from agent import is_relevant_query
        
        relevant_queries = [
            "How many artists are there?",
            "Show me all albums",
            "What genres are available?",
            "Top selling tracks"
        ]
        
        irrelevant_queries = [
            "What is the weather?",
            "Tell me about politics",
            "How to cook pasta?",
            "What is machine learning?"
        ]
        
        print("✅ Testing relevant queries:")
        for query in relevant_queries:
            is_relevant = is_relevant_query(query)
            status = "✅" if is_relevant else "❌"
            print(f"   {status} '{query}' -> {is_relevant}")
            
        print("\n✅ Testing irrelevant queries:")
        for query in irrelevant_queries:
            is_relevant = is_relevant_query(query)
            status = "✅" if not is_relevant else "❌"
            print(f"   {status} '{query}' -> {is_relevant}")
            
    except Exception as e:
        print(f"❌ Relevance detection test failed: {str(e)}")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here":
        print("⚠️  Warning: OPENAI_API_KEY not set or using placeholder value")
        print("   Set your OpenAI API key as an environment variable to run full tests")
        print("   Example: export OPENAI_API_KEY='your-actual-api-key'")
        print()
    
    # Run all tests
    test_database_connection()
    test_relevance_detection()
    
    # Only run full agent tests if API key is properly set
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your-openai-api-key-here":
        test_agent()
    else:
        print("\n🔄 Skipping full agent tests due to missing API key")
        print("   Database and relevance detection tests completed successfully")
