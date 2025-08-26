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
        initial_state = {"messages": [HumanMessage(content=query)]}

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


def test_agent_structure():
    """Test the agent structure and components without requiring API key."""
    print("\n" + "=" * 60)
    print("TESTING AGENT STRUCTURE AND COMPONENTS")
    print("=" * 60)

    try:
        # Test 1: Import and basic structure
        print("✓ Agent import successful")
        print(f"✓ Agent type: {type(app)}")

        # Test 2: Check if app has the expected methods
        if hasattr(app, "invoke"):
            print("✓ Agent has 'invoke' method")
        else:
            print("✗ Agent missing 'invoke' method")

        # Test 3: Test database initialization without full workflow
        from agent import initialize_database, extract_schema_info

        test_state = {
            "messages": [],
            "sql_query": None,
            "sql_results": None,
            "database_connection": None,
            "schema_info": None,
            "error": None,
        }

        # Test database initialization
        print("\nTesting database initialization...")
        result_state = initialize_database(test_state)

        if result_state.get("database_connection"):
            print("✓ Database initialization successful")

            # Test schema extraction
            print("Testing schema extraction...")
            schema_state = extract_schema_info(result_state)

            if schema_state.get("schema_info"):
                print("✓ Schema extraction successful")
                print(
                    f"Schema info length: {len(schema_state['schema_info'])} characters"
                )

                # Show a sample of the schema
                schema_sample = (
                    schema_state["schema_info"][:500] + "..."
                    if len(schema_state["schema_info"]) > 500
                    else schema_state["schema_info"]
                )
                print(f"Schema sample: {schema_sample}")

                # Test basic SQL query on the database
                conn = result_state["database_connection"]
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' LIMIT 5;"
                )
                tables = cursor.fetchall()
                print(
                    f"✓ Database contains {len(tables)} tables (showing first 5): "
                    f"{[t[0] for t in tables]}"
                )

                # Test a simple query
                cursor.execute("SELECT COUNT(*) FROM Artist;")
                artist_count = cursor.fetchone()[0]
                print(f"✓ Database query test: {artist_count} artists in database")

                conn.close()
            else:
                print("✗ Schema extraction failed")
        else:
            print("✗ Database initialization failed")
            if result_state.get("error"):
                print(f"Error: {result_state['error']}")

    except Exception as e:
        print(f"✗ Structure test failed: {str(e)}")


def main():
    """Run comprehensive tests of the text-to-SQL agent."""

    print("Testing LangGraph Text-to-SQL Agent")
    print("=" * 60)

    # First test the agent structure
    test_agent_structure()

    # Check if ANTHROPIC_API_KEY is set for full workflow tests
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n" + "=" * 60)
        print("ANTHROPIC_API_KEY not set - Skipping full workflow tests")
        print("=" * 60)
        print("To test the complete workflow, set the API key:")
        print("export ANTHROPIC_API_KEY='your-api-key'")
        print("Then run: python3 test_agent.py")

        print("\n" + "=" * 60)
        print("USAGE EXAMPLE")
        print("=" * 60)
        print("When the API key is set, you can use the agent like this:")
        print()
        print("from langchain_core.messages import HumanMessage")
        print("from agent import app")
        print()
        print("# Create initial state with a natural language query")
        print("initial_state = {")
        print(
            '    "messages": [HumanMessage("Who are the top 5 artists by '
            'album count?")]'
        )
        print("}")
        print()
        print("# Invoke the agent")
        print("result = app.invoke(initial_state)")
        print()
        print("# Get the response")
        print("response = result['messages'][-1].content")
        print("print(response)")
        print()
        print("The agent will:")
        print("1. Initialize the Chinook database")
        print("2. Extract schema information")
        print("3. Generate SQL from natural language")
        print("4. Execute the SQL query")
        print("5. Generate a natural language response")
        return

    print("Testing LangGraph Text-to-SQL Agent")
    print("=" * 60)

    # Test 1: Simple artist query
    test_agent_query(
        "Who are the top 5 artists by number of albums?", "Artist ranking query"
    )

    # Test 2: Customer information query
    test_agent_query("Show me customers from Brazil", "Customer filtering query")

    # Test 3: Sales analysis query
    test_agent_query("What are the total sales by country?", "Sales aggregation query")

    # Test 4: Track information query
    test_agent_query(
        "Find all rock songs longer than 5 minutes",
        "Track filtering with genre and duration",
    )

    # Test 5: Complex join query
    test_agent_query(
        "Which employee has sold the most invoices?", "Employee performance query"
    )

    # Test 6: Irrelevant query (should return "I don't know")
    test_agent_query("What's the weather like today?", "Irrelevant query test")

    # Test 7: Another irrelevant query
    test_agent_query("How do I cook pasta?", "Another irrelevant query test")

    print(f"\n{'='*60}")
    print("Testing completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


