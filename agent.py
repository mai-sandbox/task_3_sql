"""
LangGraph Text-to-SQL Agent for Chinook Database
This agent converts natural language queries to SQL, executes them against 
the Chinook database, and returns natural language responses.
"""

import sqlite3
import requests
from typing import Annotated, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# Database schema information for the Chinook database
CHINOOK_SCHEMA = """
The Chinook database contains the following tables:

1. Artist (ArtistId, Name)
   - Contains artist information

2. Album (AlbumId, Title, ArtistId)
   - Contains album information
   - Foreign key: ArtistId references Artist(ArtistId)

3. Track (TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)
   - Contains track/song information
   - Foreign keys: AlbumId references Album(AlbumId), GenreId references Genre(GenreId), MediaTypeId references MediaType(MediaTypeId)

4. Genre (GenreId, Name)
   - Contains music genre information

5. MediaType (MediaTypeId, Name)
   - Contains media type information (e.g., MPEG, AAC)

6. Customer (CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
   - Contains customer information
   - Foreign key: SupportRepId references Employee(EmployeeId)

7. Employee (EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
   - Contains employee information
   - Foreign key: ReportsTo references Employee(EmployeeId)

8. Invoice (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
   - Contains invoice information
   - Foreign key: CustomerId references Customer(CustomerId)

9. InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
   - Contains invoice line item information
   - Foreign keys: InvoiceId references Invoice(InvoiceId), TrackId references Track(TrackId)

10. Playlist (PlaylistId, Name)
    - Contains playlist information

11. PlaylistTrack (PlaylistId, TrackId)
    - Links playlists to tracks (many-to-many relationship)
    - Foreign keys: PlaylistId references Playlist(PlaylistId), TrackId references Track(TrackId)

Key relationships:
- Artist -> Album -> Track
- Customer -> Invoice -> InvoiceLine -> Track
- Track -> Genre, MediaType
- Playlist <-> Track (many-to-many via PlaylistTrack)
- Employee -> Customer (support representative)
"""


class State(TypedDict):
    """State definition for the SQL agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    sql_query: str
    sql_result: str
    final_answer: str


class ChinookDatabase:
    """Manages the in-memory Chinook SQLite database"""
    
    def __init__(self):
        self.conn = None
        self.setup_database()
    
    def setup_database(self):
        """Download and setup the Chinook database in memory"""
        try:
            # Download the Chinook SQL file
            url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
            response = requests.get(url)
            response.raise_for_status()
            sql_script = response.text
            
            # Create in-memory database
            self.conn = sqlite3.connect(":memory:")
            cursor = self.conn.cursor()
            
            # Execute the SQL script to create and populate the database
            cursor.executescript(sql_script)
            self.conn.commit()
            
            print("Chinook database loaded successfully in memory")
            
        except Exception as e:
            print(f"Error setting up database: {e}")
            raise
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results as list of dictionaries"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Fetch results
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            return results
            
        except Exception as e:
            return [{"error": str(e)}]
    
    def __del__(self):
        """Clean up database connection"""
        if self.conn:
            self.conn.close()


# Initialize the database globally
db = ChinookDatabase()


@tool
def query_chinook_database(user_query: str) -> str:
    """
    Process a natural language query about the Chinook music store database.
    This tool generates SQL, executes it, and returns a natural language response.
    
    Args:
        user_query: The user's natural language question about the music store database
    
    Returns:
        A natural language response answering the user's question
    """
    # Step 1: Check if the query is relevant to the database
    relevance_prompt = f"""Determine if the following question can be answered using a music store database 
(Chinook) that contains information about artists, albums, tracks, customers, invoices, employees, etc.

Question: {user_query}

Respond with only "RELEVANT" or "IRRELEVANT"."""
    
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    relevance_check = llm.invoke(relevance_prompt).content.strip()
    
    if "IRRELEVANT" in relevance_check.upper():
        return "I don't know the answer to that question. I can only help with queries about the music store database, including information about artists, albums, tracks, customers, invoices, and employees."
    
    # Step 2: Generate SQL query
    sql_prompt = f"""You are a SQL expert. Generate a SQL query for the Chinook database based on the user's question.

Database Schema:
{CHINOOK_SCHEMA}

User Question: {user_query}

Instructions:
1. Generate ONLY a valid SQL query - no explanations or markdown
2. Use proper JOIN clauses when querying across multiple tables
3. Use appropriate aggregate functions (COUNT, SUM, AVG, etc.) when needed
4. Limit results to 20 rows unless specifically asked for more
5. If the question cannot be answered with the available schema, return "INVALID_QUERY"

SQL Query:"""
    
    sql_response = llm.invoke(sql_prompt)
    sql_query = sql_response.content.strip()
    
    # Clean up the query if it contains markdown
    if "```sql" in sql_query.lower():
        sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql_query:
        sql_query = sql_query.split("```")[1].split("```")[0].strip()
    
    if sql_query == "INVALID_QUERY":
        return "I don't know the answer to that question based on the available database information."
    
    # Step 3: Execute the SQL query
    try:
        results = db.execute_query(sql_query)
        
        if not results:
            sql_results = "No results found for the query."
        elif "error" in results[0]:
            return f"I encountered an error while processing your query. Please try rephrasing your question."
        else:
            # Format results for the LLM
            if len(results) > 10:
                displayed_results = results[:10]
                sql_results = f"Found {len(results)} results (showing first 10):\n"
            else:
                displayed_results = results
                sql_results = f"Found {len(results)} result(s):\n"
            
            # Convert to readable format
            for row in displayed_results:
                sql_results += str(row) + "\n"
    
    except Exception as e:
        return "I encountered an error while processing your query. Please try rephrasing your question."
    
    # Step 4: Generate natural language response
    response_prompt = f"""You are a helpful assistant. Based on the SQL query results below, 
provide a clear, natural language answer to the user's question.

User Question: {user_query}

Query Results:
{sql_results}

Instructions:
1. Provide a direct, conversational answer to the user's question
2. Include specific data from the results when relevant
3. Be concise but informative
4. If the results show multiple items, summarize key findings
5. Do not mention SQL or technical details unless specifically asked

Natural Language Response:"""
    
    final_response = llm.invoke(response_prompt)
    return final_response.content.strip()


def chatbot_node(state: State) -> Dict[str, Any]:
    """
    Main chatbot node that orchestrates the SQL agent workflow.
    """
    messages = state["messages"]
    
    # Initialize the LLM with the query tool
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    llm_with_tools = llm.bind_tools([query_chinook_database])
    
    # Process the messages with the tool-enabled LLM
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


# Build the graph
def create_graph():
    """Create and compile the LangGraph workflow"""
    
    # Initialize the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("chatbot", chatbot_node)
    
    # Create tool node with our single comprehensive tool
    tools = [query_chinook_database]
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "chatbot")
    
    # Use the built-in tools_condition for proper tool routing
    workflow.add_conditional_edges(
        "chatbot",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )
    workflow.add_edge("tools", "chatbot")
    
    # Compile the graph
    return workflow.compile()


# Export the compiled graph as 'app'
app = create_graph()


# Optional: Add a simple test function
if __name__ == "__main__":
    # Test the agent with a simple query
    test_query = "How many albums are in the database?"
    initial_state = {
        "messages": [HumanMessage(content=test_query)]
    }
    
    print(f"Testing with query: {test_query}")
    result = app.invoke(initial_state)
    
    if result["messages"]:
        final_message = result["messages"][-1]
        if isinstance(final_message, AIMessage):
            print(f"Response: {final_message.content}")







