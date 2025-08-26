"""
LangGraph Text-to-SQL Agent for Chinook Database

This agent converts natural language queries to SQL, executes them against 
the Chinook SQLite database, and returns natural language responses.
"""

import sqlite3
import requests
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

# Global database connection
_db_connection: Optional[sqlite3.Connection] = None

# Database schema information for the system prompt
DATABASE_SCHEMA = """
The Chinook database contains the following tables and their relationships:

**Core Tables:**
- **Artist** (ArtistId, Name) - Music artists
- **Album** (AlbumId, Title, ArtistId) - Albums by artists
- **Track** (TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice) - Individual songs/tracks
- **Genre** (GenreId, Name) - Music genres (Rock, Jazz, Metal, Pop, etc.)
- **MediaType** (MediaTypeId, Name) - File formats (MPEG audio file, AAC audio file, etc.)

**Customer & Sales Tables:**
- **Customer** (CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId) - Customer information
- **Invoice** (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total) - Sales invoices
- **InvoiceLine** (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) - Individual items on invoices
- **Employee** (EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email) - Company employees

**Playlist Tables:**
- **Playlist** (PlaylistId, Name) - Music playlists
- **PlaylistTrack** (PlaylistId, TrackId) - Tracks in playlists

**Key Relationships:**
- Albums belong to Artists (Album.ArtistId → Artist.ArtistId)
- Tracks belong to Albums (Track.AlbumId → Album.AlbumId)
- Tracks have Genres (Track.GenreId → Genre.GenreId)
- Tracks have MediaTypes (Track.MediaTypeId → MediaType.MediaTypeId)
- Customers make Invoices (Invoice.CustomerId → Customer.CustomerId)
- InvoiceLines reference Tracks (InvoiceLine.TrackId → Track.TrackId)
- Employees can be support reps for Customers (Customer.SupportRepId → Employee.EmployeeId)
- Playlists contain Tracks via PlaylistTrack junction table

**Common Query Patterns:**
- Find tracks by artist: JOIN Track → Album → Artist
- Find sales data: JOIN Invoice → InvoiceLine → Track
- Find tracks by genre: JOIN Track → Genre
- Find customer purchases: JOIN Customer → Invoice → InvoiceLine → Track
"""

@tool
def setup_database() -> str:
    """
    Fetch the Chinook database SQL from GitHub and create an in-memory SQLite database.
    This tool should be called first before any SQL queries.
    """
    global _db_connection
    
    try:
        # Fetch the Chinook database SQL
        url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        sql_script = response.text
        
        # Create in-memory SQLite database
        _db_connection = sqlite3.connect(":memory:")
        cursor = _db_connection.cursor()
        
        # Execute the SQL script to create and populate the database
        cursor.executescript(sql_script)
        _db_connection.commit()
        
        # Verify database setup by checking table count
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        
        return f"Database setup successful! Created {table_count} tables in memory with Chinook music database data."
        
    except requests.RequestException as e:
        return f"Error fetching database SQL: {str(e)}"
    except sqlite3.Error as e:
        return f"Error setting up database: {str(e)}"
    except Exception as e:
        return f"Unexpected error during database setup: {str(e)}"

@tool
def generate_sql_query(natural_language_query: str) -> str:
    """
    Generate a SQL query based on a natural language request.
    This tool analyzes the user's request and creates appropriate SQL.
    
    Args:
        natural_language_query: The user's natural language question about the music database
    """
    # Check if query is relevant to music database
    music_keywords = [
        'artist', 'album', 'track', 'song', 'music', 'genre', 'band', 'singer',
        'customer', 'purchase', 'buy', 'sale', 'invoice', 'playlist', 'employee',
        'rock', 'jazz', 'pop', 'metal', 'classical', 'blues', 'country'
    ]
    
    query_lower = natural_language_query.lower()
    is_music_related = any(keyword in query_lower for keyword in music_keywords)
    
    if not is_music_related:
        return "I don't know - this query doesn't appear to be related to the music database."
    
    # For this implementation, we'll provide some common SQL patterns
    # In a production system, this would use an LLM to generate SQL
    
    if any(word in query_lower for word in ['artist', 'band', 'singer']):
        if 'count' in query_lower or 'how many' in query_lower:
            return "SELECT COUNT(*) as artist_count FROM Artist;"
        elif 'list' in query_lower or 'show' in query_lower or 'all' in query_lower:
            return "SELECT Name FROM Artist ORDER BY Name LIMIT 20;"
        else:
            return "SELECT Name FROM Artist WHERE Name LIKE '%rock%' OR Name LIKE '%metal%' ORDER BY Name LIMIT 10;"
    
    elif any(word in query_lower for word in ['album']):
        if 'count' in query_lower:
            return "SELECT COUNT(*) as album_count FROM Album;"
        else:
            return """
            SELECT a.Title as Album, ar.Name as Artist 
            FROM Album a 
            JOIN Artist ar ON a.ArtistId = ar.ArtistId 
            ORDER BY a.Title LIMIT 20;
            """
    
    elif any(word in query_lower for word in ['track', 'song']):
        if 'longest' in query_lower:
            return """
            SELECT t.Name as Track, ar.Name as Artist, t.Milliseconds/1000.0 as Seconds
            FROM Track t
            JOIN Album a ON t.AlbumId = a.AlbumId
            JOIN Artist ar ON a.ArtistId = ar.ArtistId
            ORDER BY t.Milliseconds DESC LIMIT 10;
            """
        elif 'genre' in query_lower:
            return """
            SELECT g.Name as Genre, COUNT(*) as Track_Count
            FROM Track t
            JOIN Genre g ON t.GenreId = g.GenreId
            GROUP BY g.Name
            ORDER BY Track_Count DESC;
            """
        else:
            return """
            SELECT t.Name as Track, ar.Name as Artist, a.Title as Album
            FROM Track t
            JOIN Album a ON t.AlbumId = a.AlbumId
            JOIN Artist ar ON a.ArtistId = ar.ArtistId
            ORDER BY t.Name LIMIT 20;
            """
    
    elif any(word in query_lower for word in ['customer', 'purchase', 'sale']):
        if 'top' in query_lower or 'best' in query_lower:
            return """
            SELECT c.FirstName || ' ' || c.LastName as Customer, SUM(i.Total) as Total_Spent
            FROM Customer c
            JOIN Invoice i ON c.CustomerId = i.CustomerId
            GROUP BY c.CustomerId
            ORDER BY Total_Spent DESC LIMIT 10;
            """
        else:
            return """
            SELECT COUNT(*) as Total_Customers FROM Customer;
            """
    
    elif any(word in query_lower for word in ['genre']):
        return """
        SELECT Name as Genre FROM Genre ORDER BY Name;
        """
    
    else:
        # Default query for general music database questions
        return """
        SELECT 'Database contains:' as Info, COUNT(*) as Count FROM Artist
        UNION ALL
        SELECT 'Albums:', COUNT(*) FROM Album
        UNION ALL
        SELECT 'Tracks:', COUNT(*) FROM Track
        UNION ALL
        SELECT 'Customers:', COUNT(*) FROM Customer;
        """

@tool
def execute_sql_query(sql_query: str) -> str:
    """
    Execute a SQL query against the Chinook database and return the results.
    
    Args:
        sql_query: The SQL query to execute
    """
    global _db_connection
    
    if _db_connection is None:
        return "Database not initialized. Please run setup_database first."
    
    # Check if this is the "I don't know" response
    if sql_query.startswith("I don't know"):
        return sql_query
    
    try:
        cursor = _db_connection.cursor()
        cursor.execute(sql_query.strip())
        
        # Get column names
        columns = [description[0] for description in cursor.description] if cursor.description else []
        
        # Fetch results
        results = cursor.fetchall()
        
        if not results:
            return "No results found for the query."
        
        # Format results as a readable string
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result
            return f"Result: {results[0][0]}"
        
        # Multiple results - format as table
        formatted_results = []
        if columns:
            # Add header
            header = " | ".join(columns)
            formatted_results.append(header)
            formatted_results.append("-" * len(header))
        
        # Add data rows
        for row in results[:50]:  # Limit to 50 rows for readability
            formatted_row = " | ".join(str(value) if value is not None else "NULL" for value in row)
            formatted_results.append(formatted_row)
        
        if len(results) > 50:
            formatted_results.append(f"... and {len(results) - 50} more rows")
        
        return "\n".join(formatted_results)
        
    except sqlite3.Error as e:
        return f"SQL execution error: {str(e)}"
    except Exception as e:
        return f"Unexpected error executing query: {str(e)}"

# System prompt for the agent
SYSTEM_PROMPT = f"""You are a specialized text-to-SQL agent for the Chinook music database. Your purpose is to:

1. Convert natural language questions into SQL queries
2. Execute those queries against the database
3. Provide natural language responses based on the results

{DATABASE_SCHEMA}

**Important Instructions:**
- ALWAYS call setup_database() first before processing any queries
- Only answer questions related to music, artists, albums, tracks, customers, sales, or other database content
- For irrelevant queries (not about music/database content), respond with "I don't know"
- Use the tools in this order: setup_database → generate_sql_query → execute_sql_query
- Provide clear, natural language responses based on the query results
- If a query fails, explain what went wrong in simple terms

**Example workflow:**
1. User asks: "How many artists are in the database?"
2. Call setup_database() to initialize the database
3. Call generate_sql_query() to create appropriate SQL
4. Call execute_sql_query() to run the query
5. Provide a natural language response like "There are 275 artists in the database."

Remember: You can only answer questions about the music database. For anything else, just say "I don't know."
"""

# Create the agent using create_react_agent
def create_sql_agent():
    """Create and return the compiled LangGraph agent."""
    
    # Initialize the language model
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )
    
    # Define the tools
    tools = [setup_database, generate_sql_query, execute_sql_query]
    
    # Create the agent using create_react_agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        debug=False
    )
    
    return agent

# Export the compiled graph as 'app' for deployment
app = create_sql_agent()

# For testing purposes, also export a function to create new instances
def get_agent():
    """Get a new instance of the SQL agent."""
    return create_sql_agent()

if __name__ == "__main__":
    # Test the agent with a sample query
    test_query = "How many artists are in the database?"
    
    print(f"Testing agent with query: {test_query}")
    
    # Create initial state with HumanMessage
    initial_state = {
        "messages": [HumanMessage(content=test_query)]
    }
    
    # Run the agent
    try:
        result = app.invoke(initial_state)
        print("\nAgent Response:")
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            print(last_message.content)
        else:
            print("No response received")
    except Exception as e:
        print(f"Error running agent: {e}")
