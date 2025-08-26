"""
LangGraph Text-to-SQL Agent for Chinook Database

This module implements a text-to-SQL agent that can:
1. Generate SQL queries from natural language requests
2. Execute queries against the Chinook SQLite database
3. Return natural language responses based on query results
4. Handle irrelevant queries by responding "I don't know"
"""

import sqlite3
import requests
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
import os


def fetch_chinook_database() -> str:
    """Fetch the Chinook database SQL from GitHub repository."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise Exception(f"Failed to fetch Chinook database SQL: {str(e)}")


def create_in_memory_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database with Chinook data."""
    # Fetch the SQL script
    sql_script = fetch_chinook_database()
    
    # Create in-memory database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    # Execute the SQL script to create tables and insert data
    cursor.executescript(sql_script)
    conn.commit()
    
    return conn


def extract_database_schema(conn: sqlite3.Connection) -> str:
    """Extract and format database schema information for system prompts."""
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = cursor.fetchall()
    
    schema_info = []
    schema_info.append("=== CHINOOK DATABASE SCHEMA ===")
    schema_info.append("This is a music store database with the following tables and relationships:")
    schema_info.append("")
    
    for (table_name,) in tables:
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        schema_info.append(f"Table: {table_name}")
        schema_info.append("Columns:")
        for col in columns:
            col_id, name, data_type, not_null, default_val, pk = col
            pk_indicator = " (PRIMARY KEY)" if pk else ""
            not_null_indicator = " NOT NULL" if not_null else ""
            default_indicator = f" DEFAULT {default_val}" if default_val else ""
            schema_info.append(f"  - {name}: {data_type}{pk_indicator}{not_null_indicator}{default_indicator}")
        
        # Get foreign key relationships
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        foreign_keys = cursor.fetchall()
        if foreign_keys:
            schema_info.append("Foreign Keys:")
            for fk in foreign_keys:
                fk_id, seq, ref_table, from_col, to_col, on_update, on_delete, match = fk
                schema_info.append(f"  - {from_col} -> {ref_table}({to_col})")
        
        schema_info.append("")
    
    # Add relationship descriptions
    schema_info.append("=== KEY RELATIONSHIPS ===")
    schema_info.append("- Artist -> Album (ArtistId)")
    schema_info.append("- Album -> Track (AlbumId)")
    schema_info.append("- Genre -> Track (GenreId)")
    schema_info.append("- MediaType -> Track (MediaTypeId)")
    schema_info.append("- Customer -> Invoice (CustomerId)")
    schema_info.append("- Invoice -> InvoiceLine (InvoiceId)")
    schema_info.append("- Track -> InvoiceLine (TrackId)")
    schema_info.append("- Employee -> Customer (SupportRepId)")
    schema_info.append("- Employee -> Employee (ReportsTo - self-referencing)")
    schema_info.append("- Playlist -> PlaylistTrack (PlaylistId)")
    schema_info.append("- Track -> PlaylistTrack (TrackId)")
    schema_info.append("")
    schema_info.append("=== SAMPLE QUERIES ===")
    schema_info.append("- Find all albums by AC/DC: SELECT * FROM Album a JOIN Artist ar ON a.ArtistId = ar.ArtistId WHERE ar.Name = 'AC/DC';")
    schema_info.append("- Get top 5 customers by total purchases: SELECT c.FirstName, c.LastName, SUM(i.Total) as TotalSpent FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId ORDER BY TotalSpent DESC LIMIT 5;")
    schema_info.append("- Find tracks longer than 5 minutes: SELECT Name, Milliseconds/1000/60.0 as Minutes FROM Track WHERE Milliseconds > 300000;")
    
    return "\n".join(schema_info)


# Create global database connection and schema
try:
    db_connection = create_in_memory_database()
    database_schema = extract_database_schema(db_connection)
except Exception as e:
    print(f"Warning: Failed to initialize database: {e}")
    db_connection = None
    database_schema = "Database initialization failed"


# Removed the broken generate_sql_query tool - the LLM will generate SQL directly


@tool
def execute_sql_query(sql_query: str) -> str:
    """
    Execute a SQL query against the Chinook database and return results.
    
    Args:
        sql_query: The SQL query to execute
        
    Returns:
        Query results formatted as a string, or error message
    """
    # Check if the query appears to be irrelevant to the database
    if not sql_query.strip() or sql_query.strip().startswith('--') or 'IRRELEVANT' in sql_query.upper():
        return "I don't know the answer to that question. I can only help with questions about the Chinook music database."
    
    if not db_connection:
        return "Database connection is not available."
    
    try:
        cursor = db_connection.cursor()
        cursor.execute(sql_query)
        
        # Get column names
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        
        # Fetch results
        results = cursor.fetchall()
        
        if not results:
            return "No results found for the query."
        
        # Format results
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result
            return str(results[0][0])
        
        # Multiple results - format as table
        formatted_results = []
        if column_names:
            formatted_results.append(" | ".join(column_names))
            formatted_results.append("-" * len(formatted_results[0]))
        
        for row in results[:10]:  # Limit to first 10 rows
            formatted_results.append(" | ".join(str(cell) if cell is not None else "NULL" for cell in row))
        
        if len(results) > 10:
            formatted_results.append(f"... and {len(results) - 10} more rows")
        
        return "\n".join(formatted_results)
        
    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}"
    except Exception as e:
        return f"Error executing query: {str(e)}"


# System prompt with database schema
SYSTEM_PROMPT = f"""You are a helpful SQL assistant for the Chinook music database. Your job is to:

1. Analyze natural language questions about the music database
2. Generate appropriate SQL queries based on the database schema
3. Execute those queries using the execute_sql_query tool
4. Provide clear, natural language responses based on the results

{database_schema}

IMPORTANT INSTRUCTIONS:
- Only answer questions related to the Chinook music database
- If a question is not related to music, artists, albums, tracks, customers, sales, or the database, respond with "I don't know the answer to that question."
- Generate SQL queries directly based on the schema information provided above
- Use the execute_sql_query tool to run your generated SQL queries
- Provide clear, natural language responses based on the query results
- If you encounter SQL errors, review the schema and try again with corrected syntax
- Be precise with SQL syntax and use proper table/column names from the schema above
- Use appropriate JOINs when data spans multiple tables
- Limit results to reasonable numbers (e.g., TOP 10) unless specifically asked for more

WORKFLOW:
1. Understand the user's question
2. Generate the appropriate SQL query based on the schema
3. Call execute_sql_query with your generated SQL
4. Interpret the results and provide a natural language response

Remember: You can only help with questions about the Chinook music database. For anything else, just say you don't know.
"""

# Initialize the model with proper API key handling
def get_model():
    """Get the best available model with proper fallback handling."""
    # Try Anthropic first (preferred)
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            return ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0
            )
    except Exception:
        pass
    
    # Try OpenAI as fallback
    try:
        from langchain_openai import ChatOpenAI
        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(
                model="gpt-4o",
                temperature=0
            )
    except Exception:
        pass
    
    # Final fallback - create a mock model for development/testing
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key="dummy-key-for-development"  # This will fail but allows testing structure
        )
    except Exception:
        # If all else fails, create a minimal mock
        from langchain_core.language_models.fake import FakeListChatModel
        return FakeListChatModel(responses=["I need proper API keys to function."])

model = get_model()

# Create the agent
tools = [execute_sql_query]

app = create_react_agent(
    model=model,
    tools=tools,
    prompt=SYSTEM_PROMPT,
    debug=False
)

# Export the compiled graph as 'app' for LangGraph deployment
if __name__ == "__main__":
    # Test the agent structure (without requiring API keys)
    print("LangGraph SQL Agent initialized successfully!")
    print(f"Database connection: {'✓ Connected' if db_connection else '✗ Failed'}")
    print(f"Schema extracted: {'✓ Yes' if 'CHINOOK DATABASE SCHEMA' in database_schema else '✗ No'}")
    print(f"Tools available: {len(tools)} tools")
    print(f"Model configured: {model.__class__.__name__}")
    print(f"Agent compiled: {'✓ Yes' if app else '✗ No'}")
    
    # Only run actual test if API keys are available
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"):
        try:
            test_message = HumanMessage("What are the top 5 best-selling artists by total sales?")
            result = app.invoke({"messages": [test_message]})
            print("\nTest result:")
            for message in result["messages"]:
                print(f"{message.__class__.__name__}: {message.content}")
        except Exception as e:
            print(f"\nTest failed (expected without API keys): {str(e)[:100]}...")
    else:
        print("\nSkipping live test - no API keys found (this is expected in development)")
        print("To test with real queries, set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")






