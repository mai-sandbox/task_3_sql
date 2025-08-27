"""
Test script to demonstrate the text-to-SQL agent usage.
Note: Requires OPENAI_API_KEY environment variable to be set.
"""

from agent import app
from langchain_core.messages import HumanMessage
import os

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable to test the agent.")
    print("\nExample usage:")
    print("export OPENAI_API_KEY='your-api-key-here'")
    print("python test_agent.py")
    print("\nThe agent can answer questions like:")
    print("- How many customers are there?")
    print("- What are the top 5 best selling albums?")
    print("- Which artist has the most albums?")
    print("- List all genres in the database")
    print("- What's the total revenue from all invoices?")
    exit(1)

# Example queries to test
test_queries = [
    "How many customers are there in the database?",
    "What are the top 3 best selling albums?",
    "Which employee has been with the company the longest?",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Question: {query}")
    print('-'*60)
    
    # Invoke the agent with minimal state
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    try:
        result = app.invoke(initial_state)
        
        # Extract the final AI response
        final_message = result["messages"][-1].content
        print(f"Answer: {final_message}")
        
        # Show the generated SQL for transparency
        if "sql_query" in result and result["sql_query"]:
            print(f"\nSQL Generated: {result['sql_query']}")
    except Exception as e:
        print(f"Error: {e}")