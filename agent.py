"""
LangGraph Text-to-SQL Agent for Chinook Database

This agent converts natural language queries to SQL, executes them against 
the Chinook SQLite database, and returns results in natural language.
"""

import sqlite3
import requests
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

# Initialize the database connection (global to persist across tool calls)
_db_connection = None

def initialize_database():
    """Initialize the Chinook database from the remote SQL file."""
    global _db_connection
    
    if _db_connection is not None:
        return _db_connection
    
    try:
        # Fetch the Chinook database schema
        response = requests.get(
            "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql",
            timeout=30
        )
        response.raise_for_status()
        
        # Create in-memory SQLite database
        _db_connection = sqlite3.connect(":memory:")
        cursor = _db_connection.cursor()
        
        # Execute the SQL schema and data
        cursor.executescript(response.text)
        _db_connection.commit()
        
        return _db_connection
        
    except Exception as e:
        raise Exception(f"Failed to initialize database: {str(e)}")

@tool
def execute_sql_query(query: str) -> str:
    """
    Execute a SQL query against the Chinook database and return the results.
    
    Args:
        query: The SQL query to execute
        
    Returns:
        String representation of the query results or error message
    """
    try:
        # Initialize database if not already done
        db = initialize_database()
        cursor = db.cursor()
        
        # Execute the query
        cursor.execute(query)
        
        # Fetch results
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        
        if not results:
            return "No results found for the query."
        
        # Format results as a readable string
        formatted_results = []
        formatted_results.append(" | ".join(column_names))
        formatted_results.append("-" * len(" | ".join(column_names)))
        
        for row in results:
            formatted_results.append(" | ".join(str(value) if value is not None else "NULL" for value in row))
        
        # Limit results to prevent overwhelming output
        if len(formatted_results) > 52:  # Header + separator + 50 rows
            formatted_results = formatted_results[:52]
            formatted_results.append("... (results truncated)")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error executing query: {str(e)}"

@tool
def get_database_schema() -> str:
    """
    Get the schema information for all tables in the Chinook database.
    
    Returns:
        String containing table schemas and relationships
    """
    try:
        # Initialize database if not already done
        db = initialize_database()
        cursor = db.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cursor.fetchall()
        
        schema_info = []
        schema_info.append("CHINOOK DATABASE SCHEMA")
        schema_info.append("=" * 50)
        schema_info.append("")
        
        for (table_name,) in tables:
            schema_info.append(f"TABLE: {table_name}")
            schema_info.append("-" * (len(table_name) + 7))
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            for column in columns:
                col_id, col_name, col_type, not_null, default_val, is_pk = column
                pk_indicator = " (PRIMARY KEY)" if is_pk else ""
                null_indicator = " NOT NULL" if not_null else ""
                default_indicator = f" DEFAULT {default_val}" if default_val else ""
                
                schema_info.append(f"  {col_name}: {col_type}{pk_indicator}{null_indicator}{default_indicator}")
            
            # Get foreign key information
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = cursor.fetchall()
            
            if foreign_keys:
                schema_info.append("  Foreign Keys:")
                for fk in foreign_keys:
                    fk_id, seq, ref_table, from_col, to_col, on_update, on_delete, match = fk
                    schema_info.append(f"    {from_col} -> {ref_table}({to_col})")
            
            schema_info.append("")
        
        # Add relationship summary
        schema_info.append("KEY RELATIONSHIPS:")
        schema_info.append("-" * 17)
        schema_info.append("• Album -> Artist (ArtistId)")
        schema_info.append("• Track -> Album (AlbumId)")
        schema_info.append("• Track -> Genre (GenreId)")
        schema_info.append("• Track -> MediaType (MediaTypeId)")
        schema_info.append("• Customer -> Employee (SupportRepId)")
        schema_info.append("• Invoice -> Customer (CustomerId)")
        schema_info.append("• InvoiceLine -> Invoice (InvoiceId)")
        schema_info.append("• InvoiceLine -> Track (TrackId)")
        schema_info.append("• PlaylistTrack -> Playlist (PlaylistId)")
        schema_info.append("• PlaylistTrack -> Track (TrackId)")
        schema_info.append("")
        schema_info.append("This is a music store database containing information about:")
        schema_info.append("- Artists and their albums")
        schema_info.append("- Tracks with genre and media type information")
        schema_info.append("- Customers and their purchase history")
        schema_info.append("- Employees and customer support relationships")
        schema_info.append("- Playlists and track associations")
        
        return "\n".join(schema_info)
        
    except Exception as e:
        return f"Error retrieving schema: {str(e)}"

# System prompt with comprehensive database context
SYSTEM_PROMPT = """You are a specialized text-to-SQL agent for the Chinook music database. Your role is to:

1. Convert natural language questions into accurate SQL queries
2. Execute those queries against the database
3. Interpret the results and provide clear, natural language responses

IMPORTANT GUIDELINES:
- Only answer questions that can be resolved using the Chinook database
- If a question is irrelevant to the music database or cannot be answered with the available data, respond with "I don't know the answer"
- Always use the get_database_schema tool first if you need to understand the database structure
- Use the execute_sql_query tool to run your SQL queries
- Provide clear, conversational responses based on the query results
- Be precise with SQL syntax and table/column names

The Chinook database contains music store data with these main entities:
- Artists and Albums
- Tracks with genres and media types  
- Customers and their purchase history
- Employees and customer support
- Playlists and track associations

When generating SQL:
- Use proper JOIN syntax for relationships
- Be mindful of case sensitivity in column names
- Use appropriate WHERE clauses for filtering
- Consider using LIMIT for large result sets
- Handle NULL values appropriately

Remember: Your purpose is solely to help users query and understand the Chinook music database. Stay focused on this domain."""

# Initialize the model
model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0
)

# Create the agent with tools
agent = create_react_agent(
    model=model,
    tools=[execute_sql_query, get_database_schema],
    prompt=SYSTEM_PROMPT
)

# Export the compiled graph as 'app' for deployment
app = agent

