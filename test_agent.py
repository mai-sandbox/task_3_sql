#!/usr/bin/env python3
"""
Test script for the LangGraph Text-to-SQL Agent

This script tests the complete workflow including:
- Natural language input processing
- SQL generation
- SQL execution against Chinook database
- Natural language response generation
"""

import os
from langchain_core.messages import HumanMessage
from agent import app

def test_agent_query(query: str, description: str):
    """Test a single query against the agent."""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print("-" * 60)
    
    try:
        # Create initial state with the human message
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Invoke the agent
        result = app.invoke(initial_state)
        
        # Extract the final response
        if result.get("messages"):
            final_message = result["messages"][-1]
            print(f"Response: {final_message.content}")
            
            # Show additional debug info if available
            if result.get("sql_query"):
                print(f"\nGenerated SQL: {result['sql_query']}")
            if result.get("sql_results") and len(result["sql_results"]) > 0:
                print(f"Results count: {len(result['sql_results'])} rows")
                if len(result["sql_results"]) <= 3:  # Show first few results
                    print(f"Sample results: {result['sql_results']}")
        else:
            print("No response received")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
    
    print("-" * 60)

def main():
    """Run comprehensive tests of the text-to-SQL agent."""
    
    # Check if ANTHROPIC_API_KEY is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY environment variable not set.")
        print("The agent may not work properly without it.")
        print("Please set it with: export ANTHROPIC_API_KEY='your-api-key'")
        return
    
    print("Testing LangGraph Text-to-SQL Agent")
    print("=" * 60)
    
    # Test 1: Simple artist query
    test_agent_query(
        "Who are the top 5 artists by number of albums?",
        "Artist ranking query"
    )
    
    # Test 2: Customer information query
    test_agent_query(
        "Show me customers from Brazil",
        "Customer filtering query"
    )
    
    # Test 3: Sales analysis query
    test_agent_query(
        "What are the total sales by country?",
        "Sales aggregation query"
    )
    
    # Test 4: Track information query
    test_agent_query(
        "Find all rock songs longer than 5 minutes",
        "Track filtering with genre and duration"
    )
    
    # Test 5: Complex join query
    test_agent_query(
        "Which employee has sold the most invoices?",
        "Employee performance query"
    )
    
    # Test 6: Irrelevant query (should return "I don't know")
    test_agent_query(
        "What's the weather like today?",
        "Irrelevant query test"
    )
    
    # Test 7: Another irrelevant query
    test_agent_query(
        "How do I cook pasta?",
        "Another irrelevant query test"
    )
    
    print(f"\n{'='*60}")
    print("Testing completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
