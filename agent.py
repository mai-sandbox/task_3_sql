"""
LangGraph Text-to-SQL Agent for Chinook Database

This agent converts natural language queries to SQL, executes them against 
the Chinook SQLite database, and returns natural language responses.
"""

import sqlite3
import requests
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
import os


# State schema for LangGraph workflow
class AgentState(TypedDict):
    messages: List[BaseMessage]
    sql_query: str
    sql_result: str
    error: str


# Initialize in-memory SQLite database with Chinook schema
def initialize_database():
    """Fetch Chinook schema from URL and initialize in-memory SQLite database."""
    try:
        # Fetch the Chinook database schema
        response = requests.get(
            "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
        )
        response.raise_for_status()
        
        # Create in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # Execute the schema SQL
        cursor.executescript(response.text)
        conn.commit()
        
        return conn
    except Exception as e:
        raise Exception(f"Failed to initialize database: {str(e)}")


# Global database connection
db_connection = initialize_database()


# Database schema information for LLM context
DATABASE_SCHEMA = """
The Chinook database contains the following tables and their relationships:

1. **Artist** (ArtistId, Name)
   - Contains music artists

2. **Album** (AlbumId, Title, ArtistId)
   - Contains albums, linked to artists
   - Foreign key: ArtistId -> Artist.ArtistId

3. **Track** (TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)
   - Contains individual tracks/songs
   - Foreign keys: AlbumId -> Album.AlbumId, MediaTypeId -> MediaType.MediaTypeId, GenreId -> Genre.GenreId

4. **Genre** (GenreId, Name)
   - Contains music genres (Rock, Jazz, Metal, etc.)

5. **MediaType** (MediaTypeId, Name)
   - Contains media types (MPEG audio file, AAC audio file, etc.)

6. **Customer** (CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
   - Contains customer information
   - Foreign key: SupportRepId -> Employee.EmployeeId

7. **Employee** (EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
   - Contains employee information
   - Foreign key: ReportsTo -> Employee.EmployeeId (self-referencing)

8. **Invoice** (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
   - Contains invoice headers
   - Foreign key: CustomerId -> Customer.CustomerId

9. **InvoiceLine** (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
   - Contains invoice line items
   - Foreign keys: InvoiceId -> Invoice.InvoiceId, TrackId -> Track.TrackId

10. **Playlist** (PlaylistId, Name)
    - Contains playlist information

11. **PlaylistTrack** (PlaylistId, TrackId)
    - Many-to-many relationship between playlists and tracks
    - Foreign keys: PlaylistId -> Playlist.PlaylistId, TrackId -> Track.TrackId

Key relationships:
- Artist -> Album -> Track (music hierarchy)
- Customer -> Invoice -> InvoiceLine -> Track (sales data)
- Genre/MediaType -> Track (track categorization)
- Playlist -> PlaylistTrack -> Track (playlist management)
"""


# Initialize LLM
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def generate_sql(state: AgentState) -> AgentState:
    """Convert natural language query to SQL using LLM with schema context."""
    try:
        # Get the latest human message
        human_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                human_message = msg.content
                break
        
        if not human_message:
            state["error"] = "No human message found"
            return state
        
        # Create prompt with schema context
        prompt = f"""You are a SQL expert working with the Chinook music database. 
        
{DATABASE_SCHEMA}

Convert the following natural language query to a valid SQLite SQL query. 

Rules:
1. Only generate SELECT queries - no INSERT, UPDATE, DELETE, or DDL statements
2. Use proper SQLite syntax
3. Include appropriate JOINs when querying multiple tables
4. Use LIMIT clause for queries that might return many results (limit to 10 unless specifically asked for more)
5. If the query is not related to music, artists, albums, tracks, customers, invoices, employees, or playlists, respond with "IRRELEVANT_QUERY"
6. Return only the SQL query, no explanations or formatting

Human query: {human_message}

SQL Query:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        sql_query = response.content.strip()
        
        # Check if query is irrelevant
        if "IRRELEVANT_QUERY" in sql_query:
            state["error"] = "irrelevant_query"
            return state
        
        # Clean up the SQL query (remove any markdown formatting)
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
        
        state["sql_query"] = sql_query
        return state
        
    except Exception as e:
        state["error"] = f"SQL generation error: {str(e)}"
        return state


def execute_sql(state: AgentState) -> AgentState:
    """Execute SQL query against the database and handle errors."""
    try:
        # Check if there's an error from previous step
        if state.get("error"):
            return state
        
        sql_query = state.get("sql_query", "")
        if not sql_query:
            state["error"] = "No SQL query to execute"
            return state
        
        # Execute the SQL query
        cursor = db_connection.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        
        # Format results
        if not results:
            state["sql_result"] = "No results found."
        else:
            # Convert results to a readable format
            formatted_results = []
            for row in results:
                row_dict = dict(zip(column_names, row))
                formatted_results.append(row_dict)
            
            state["sql_result"] = str(formatted_results)
        
        return state
        
    except sqlite3.Error as e:
        state["error"] = f"SQL execution error: {str(e)}"
        return state
    except Exception as e:
        state["error"] = f"Unexpected error during SQL execution: {str(e)}"
        return state


def generate_response(state: AgentState) -> AgentState:
    """Generate natural language response from SQL results."""
    try:
        # Handle errors from previous steps
        if state.get("error"):
            if state["error"] == "irrelevant_query":
                response = "I don't know the answer to that question. I can only help with queries related to the music database (artists, albums, tracks, customers, invoices, employees, and playlists)."
            else:
                response = "I don't know the answer. There was an error processing your query."
            
            state["messages"].append(AIMessage(content=response))
            return state
        
        # Get the original human query
        human_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                human_message = msg.content
                break
        
        sql_query = state.get("sql_query", "")
        sql_result = state.get("sql_result", "")
        
        # Create prompt for natural language response
        prompt = f"""Based on the SQL query results, provide a clear and natural language response to the user's question.

Original question: {human_message}
SQL query executed: {sql_query}
Query results: {sql_result}

Rules:
1. Provide a clear, conversational response
2. Include specific details from the results when relevant
3. If no results were found, say so clearly
4. Don't mention the SQL query or technical details
5. Keep the response concise but informative

Response:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        
        state["messages"].append(AIMessage(content=response.content))
        return state
        
    except Exception as e:
        error_response = "I don't know the answer. There was an error generating the response."
        state["messages"].append(AIMessage(content=error_response))
        return state


# Build the LangGraph workflow
def build_graph():
    """Build and return the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("generate_response", generate_response)
    
    # Define the workflow edges
    workflow.set_entry_point("generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()


# Export the compiled graph as required
app = build_graph()
