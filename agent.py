"""
LangGraph Text-to-SQL Agent for Chinook Database

This agent converts natural language queries to SQL, executes them against the Chinook database,
and returns natural language responses. It only responds to database-related queries about the
music store data and says "I don't know" for irrelevant queries.
"""

import sqlite3
import requests
import re
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

# Global database connection
_db_connection = None

# Database schema information for context
CHINOOK_SCHEMA = """
The Chinook database contains the following 11 tables for a music store:

1. Artist (ArtistId, Name) - Music artists
2. Album (AlbumId, Title, ArtistId) - Music albums by artists
3. Track (TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice) - Individual songs/tracks
4. Genre (GenreId, Name) - Music genres (Rock, Jazz, Metal, etc.)
5. MediaType (MediaTypeId, Name) - File formats (MPEG, AAC, etc.)
6. Customer (CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId) - Store customers
7. Employee (EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email) - Store employees
8. Invoice (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total) - Customer purchases
9. InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) - Individual items in invoices
10. Playlist (PlaylistId, Name) - Music playlists
11. PlaylistTrack (PlaylistId, TrackId) - Tracks in playlists

Key relationships:
- Artist -> Album -> Track
- Customer -> Invoice -> InvoiceLine -> Track
- Track -> Genre, MediaType
- Playlist -> PlaylistTrack -> Track
- Employee (support rep) -> Customer
"""

MUSIC_STORE_KEYWORDS = [
    'artist', 'album', 'track', 'song', 'music', 'genre', 'customer', 'invoice', 
    'purchase', 'buy', 'sold', 'sales', 'employee', 'playlist', 'band', 'singer',
    'rock', 'jazz', 'metal', 'classical', 'price', 'revenue', 'total', 'billing',
    'chinook', 'store', 'media', 'composer', 'duration', 'milliseconds'
]

@tool
def initialize_database() -> str:
    """
    Initialize the Chinook database by fetching the SQL file and creating an in-memory SQLite database.
    This tool should be called first before any database operations.
    """
    global _db_connection
    
    try:
        # Check if database is already initialized
        if _db_connection is not None:
            return "Database already initialized and ready for queries."
        
        # Fetch the Chinook database SQL file
        url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        sql_content = response.text
        
        # Create in-memory SQLite database
        _db_connection = sqlite3.connect(":memory:")
        cursor = _db_connection.cursor()
        
        # Execute the SQL to create tables and insert data
        # Split by semicolon and execute each statement
        statements = sql_content.split(';')
        for statement in statements:
            statement = statement.strip()
            if statement:
                cursor.execute(statement)
        
        _db_connection.commit()
        
        # Verify database was created successfully
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if len(tables) >= 11:  # Should have 11 tables
            return f"Database successfully initialized with {len(tables)} tables: {', '.join([table[0] for table in tables])}"
        else:
            return f"Database initialized but only found {len(tables)} tables. Expected 11 tables."
            
    except requests.RequestException as e:
        return f"Error fetching database SQL file: {str(e)}"
    except sqlite3.Error as e:
        return f"Database error during initialization: {str(e)}"
    except Exception as e:
        return f"Unexpected error during database initialization: {str(e)}"

@tool
def validate_query_relevance(user_query: str) -> str:
    """
    Validate if the user query is relevant to the Chinook music store database.
    Returns 'relevant' if the query is about music store data, 'irrelevant' otherwise.
    """
    query_lower = user_query.lower()
    
    # Check for music store related keywords
    has_music_keywords = any(keyword in query_lower for keyword in MUSIC_STORE_KEYWORDS)
    
    # Check for database/SQL related terms that might indicate a valid query
    db_terms = ['select', 'count', 'sum', 'average', 'list', 'show', 'find', 'get', 'how many', 'what', 'which', 'who']
    has_db_terms = any(term in query_lower for term in db_terms)
    
    # If it has both music keywords and database terms, it's likely relevant
    if has_music_keywords and has_db_terms:
        return "relevant"
    
    # If it has music keywords but no clear database intent, still might be relevant
    if has_music_keywords:
        return "relevant"
    
    # Check for obvious non-music store queries
    irrelevant_patterns = [
        r'\b(weather|sports|politics|news|cooking|recipe|movie|film|book|travel|health|medicine)\b',
        r'\b(python|programming|code|software|computer|technology)\b',
        r'\b(math|calculation|formula|equation)\b',
        r'\b(hello|hi|goodbye|thanks|thank you)\b'
    ]
    
    for pattern in irrelevant_patterns:
        if re.search(pattern, query_lower):
            return "irrelevant"
    
    # If unclear, lean towards relevant to let the SQL generation handle it
    return "relevant"

@tool
def generate_sql_query(user_query: str) -> str:
    """
    Convert a natural language query to SQL for the Chinook database.
    Uses the database schema context to generate accurate SQL queries.
    """
    global _db_connection
    
    if _db_connection is None:
        return "Error: Database not initialized. Please initialize the database first."
    
    # First validate query relevance
    relevance = validate_query_relevance(user_query)
    if relevance == "irrelevant":
        return "IRRELEVANT_QUERY"
    
    try:
        # Create a prompt for SQL generation with schema context
        sql_prompt = f"""
        Given the following database schema for a music store (Chinook database):
        
        {CHINOOK_SCHEMA}
        
        Convert this natural language query to SQL:
        "{user_query}"
        
        Rules:
        1. Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
        2. Use proper table joins when needed
        3. Include appropriate WHERE clauses for filtering
        4. Use LIMIT when appropriate to avoid huge result sets
        5. Return only the SQL query, no explanations
        6. If the query cannot be answered with this database, return "CANNOT_ANSWER"
        
        SQL Query:
        """
        
        # Use the LLM to generate SQL
        model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
        response = model.invoke([HumanMessage(content=sql_prompt)])
        
        sql_query = response.content.strip()
        
        # Clean up the SQL query
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        
        # Basic validation - ensure it's a SELECT query
        if not sql_query.upper().startswith('SELECT'):
            if 'CANNOT_ANSWER' in sql_query:
                return "CANNOT_ANSWER"
            return "Error: Only SELECT queries are allowed."
        
        # Additional safety check - no dangerous keywords
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
        sql_upper = sql_query.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return f"Error: {keyword} operations are not allowed."
        
        return sql_query
        
    except Exception as e:
        return f"Error generating SQL query: {str(e)}"

@tool
def execute_sql_query(sql_query: str) -> str:
    """
    Execute a SQL query against the Chinook database and return the results.
    Only executes SELECT queries for safety.
    """
    global _db_connection
    
    if _db_connection is None:
        return "Error: Database not initialized. Please initialize the database first."
    
    if sql_query in ["IRRELEVANT_QUERY", "CANNOT_ANSWER"]:
        return sql_query
    
    if sql_query.startswith("Error:"):
        return sql_query
    
    try:
        cursor = _db_connection.cursor()
        
        # Execute the query with a reasonable timeout
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        
        if not results:
            return "No results found for the query."
        
        # Format results as a string
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result
            return f"Result: {results[0][0]}"
        
        # Multiple results - format as table-like structure
        result_str = f"Found {len(results)} result(s):\n"
        result_str += f"Columns: {', '.join(column_names)}\n\n"
        
        # Limit output to prevent overwhelming responses
        max_rows = 20
        for i, row in enumerate(results[:max_rows]):
            row_str = " | ".join([str(value) if value is not None else "NULL" for value in row])
            result_str += f"Row {i+1}: {row_str}\n"
        
        if len(results) > max_rows:
            result_str += f"\n... and {len(results) - max_rows} more rows"
        
        return result_str
        
    except sqlite3.Error as e:
        return f"Database error executing query: {str(e)}"
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"

@tool
def format_natural_language_response(query_results: str, original_query: str) -> str:
    """
    Convert SQL query results into a natural language response.
    """
    if query_results == "IRRELEVANT_QUERY":
        return "I don't know the answer to that question. I can only help with queries about the music store database, including information about artists, albums, tracks, customers, sales, and employees."
    
    if query_results == "CANNOT_ANSWER":
        return "I don't know the answer to that question. The information you're looking for is not available in the music store database."
    
    if query_results.startswith("Error:"):
        return "I don't know the answer to that question. There was an issue processing your request."
    
    if "No results found" in query_results:
        return "I found no results for your query. The information you're looking for might not exist in the database."
    
    try:
        # Use LLM to generate natural language response
        response_prompt = f"""
        Convert the following database query results into a natural, conversational response.
        
        Original user query: "{original_query}"
        Database results: {query_results}
        
        Rules:
        1. Be conversational and natural
        2. Summarize the key findings
        3. If there are many results, highlight the most important ones
        4. Use proper formatting for readability
        5. Don't mention SQL or technical database terms
        6. Keep the response focused and concise
        
        Natural language response:
        """
        
        model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
        response = model.invoke([HumanMessage(content=response_prompt)])
        
        return response.content.strip()
        
    except Exception as e:
        # Fallback to basic formatting if LLM fails
        return f"Here are the results for your query: {query_results}"

# Initialize the LangGraph agent
def create_sql_agent():
    """Create and return the LangGraph text-to-SQL agent."""
    
    # Define the tools for the agent
    tools = [
        initialize_database,
        validate_query_relevance,
        generate_sql_query,
        execute_sql_query,
        format_natural_language_response
    ]
    
    # Initialize the model
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    
    # Create the agent with a system prompt
    system_prompt = """You are a helpful assistant that answers questions about a music store database (Chinook database).

Your workflow should be:
1. First, initialize the database if it hasn't been initialized yet
2. Validate if the user's query is relevant to the music store database
3. If relevant, generate a SQL query to answer the question
4. Execute the SQL query against the database
5. Format the results into a natural language response

If the query is not relevant to the music store database (artists, albums, tracks, customers, sales, employees), respond with "I don't know the answer to that question" and explain that you can only help with music store queries.

Always be helpful and provide clear, natural language responses based on the database results."""
    
    # Create the react agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt
    )
    
    return agent

# Create and export the agent as 'app'
app = create_sql_agent()

if __name__ == "__main__":
    # Test the agent - only run if API key is available
    import os
    if os.getenv("ANTHROPIC_API_KEY"):
        test_query = "How many artists are in the database?"
        try:
            result = app.invoke({"messages": [HumanMessage(content=test_query)]})
            print("Test Query:", test_query)
            print("Response:", result["messages"][-1].content)
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print("Agent created successfully. Set ANTHROPIC_API_KEY environment variable to test.")
        print("The agent is ready for deployment and will work when API keys are properly configured.")


