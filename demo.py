"""
Demonstration of the Text-to-SQL Agent Structure

This script shows how to use the agent once you have set up your ANTHROPIC_API_KEY
in a .env file.
"""

from langchain_core.messages import HumanMessage

print("Text-to-SQL Agent Demo")
print("=" * 50)
print("\nTo use this agent:")
print("1. Create a .env file with your ANTHROPIC_API_KEY")
print("2. Import the app from agent.py")
print("3. Invoke with a message state\n")

print("Example usage:")
print("-" * 30)
print("""
from agent import app
from langchain_core.messages import HumanMessage

# Create initial state with your question
initial_state = {
    "messages": [HumanMessage("How many albums are in the database?")]
}

# Invoke the agent
result = app.invoke(initial_state)

# Get the response
response = result["messages"][-1].content
print(response)
""")

print("\nThe agent workflow:")
print("-" * 30)
print("1. Accepts natural language query")
print("2. Generates SQL query using the Chinook database schema")
print("3. Executes the SQL against an in-memory SQLite database")
print("4. Converts results to natural language response")
print("5. Returns 'I don't know' for irrelevant queries")

print("\nSupported query examples:")
print("-" * 30)
queries = [
    "How many customers are there?",
    "What are the top selling tracks?",
    "List albums by a specific artist",
    "Show employee information",
    "What genres are available?",
    "Total sales by country"
]

for q in queries:
    print(f"  • {q}")

print("\nThe Chinook database includes tables for:")
print("-" * 30)
tables = [
    "Albums", "Artists", "Customers", "Employees",
    "Genres", "Invoices", "InvoiceItems", "MediaTypes",
    "Playlists", "PlaylistTrack", "Tracks"
]

for t in tables:
    print(f"  • {t}")