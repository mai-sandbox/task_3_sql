"""
LangGraph Text-to-SQL Agent for Chinook Database
This agent converts natural language queries to SQL, executes them against 
the Chinook database, and returns natural language responses.
"""

import sqlite3
import requests
from typing import Annotated, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
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
def generate_sql_query(user_query: str) -> str:
    """
    Generate a SQL query based on the user's natural language question.
    This tool analyzes the user's request and creates an appropriate SQL query
    for the Chinook database.
    
    Args:
        user_query: The user's natural language question about the database
    
    Returns:
        A SQL query string
    """
    # Create a prompt with schema information
    prompt = f"""You are a SQL expert. Generate a SQL query for the Chinook database based on the user's question.

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
    
    # Use the LLM to generate SQL
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    response = llm.invoke(prompt)
    
    sql_query = response.content.strip()
    
    # Clean up the query if it contains markdown
    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql_query:
        sql_query = sql_query.split("```")[1].split("```")[0].strip()
    
    return sql_query


@tool
def execute_sql_query(sql_query: str) -> str:
    """
    Execute a SQL query against the Chinook database and return the results.
    
    Args:
        sql_query: The SQL query to execute
    
    Returns:
        A string representation of the query results or error message
    """
    if sql_query == "INVALID_QUERY":
        return "I don't know the answer to that question based on the available database information."
    
    try:
        results = db.execute_query(sql_query)
        
        if not results:
            return "No results found for the query."
        
        if "error" in results[0]:
            return f"Error executing query: {results[0]['error']}"
        
        # Format results as a readable string
        if len(results) > 10:
            displayed_results = results[:10]
            result_str = f"Showing first 10 of {len(results)} results:\n\n"
        else:
            displayed_results = results
            result_str = f"Found {len(results)} result(s):\n\n"
        
        # Convert results to readable format
        for i, row in enumerate(displayed_results, 1):
            result_str += f"Result {i}:\n"
            for key, value in row.items():
                result_str += f"  {key}: {value}\n"
            result_str += "\n"
        
        return result_str
        
    except Exception as e:
        return f"Error executing query: {str(e)}"


@tool
def generate_natural_language_response(user_query: str, sql_query: str, sql_results: str) -> str:
    """
    Generate a natural language response based on the SQL query results.
    
    Args:
        user_query: The original user question
        sql_query: The SQL query that was executed
        sql_results: The results from executing the SQL query
    
    Returns:
        A natural language response answering the user's question
    """
    if "I don't know" in sql_results or "Error" in sql_results:
        return sql_results
    
    prompt = f"""You are a helpful assistant. Based on the SQL query results below, 
provide a clear, natural language answer to the user's question.

User Question: {user_query}

SQL Query Executed: {sql_query}

Query Results:
{sql_results}

Instructions:
1. Provide a direct, conversational answer to the user's question
2. Include specific data from the results when relevant
3. Be concise but informative
4. If the results show multiple items, summarize key findings
5. Do not mention SQL or technical details unless specifically asked

Natural Language Response:"""
    
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    response = llm.invoke(prompt)
    
    return response.content.strip()


def chatbot_node(state: State) -> Dict[str, Any]:
    """
    Main chatbot node that orchestrates the SQL agent workflow.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if this is a user message
    if isinstance(last_message, HumanMessage):
        user_query = last_message.content
        
        # Check if the query is relevant to the database
        relevance_prompt = f"""Determine if the following question can be answered using a music store database 
(Chinook) that contains information about artists, albums, tracks, customers, invoices, employees, etc.

Question: {user_query}

Respond with only "RELEVANT" or "IRRELEVANT"."""
        
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
        relevance_check = llm.invoke(relevance_prompt).content.strip()
        
        if "IRRELEVANT" in relevance_check.upper():
            response = "I don't know the answer to that question. I can only help with queries about the music store database, including information about artists, albums, tracks, customers, invoices, and employees."
            return {"messages": [AIMessage(content=response)]}
        
        # Use the LLM with tools to process the query
        llm_with_tools = llm.bind_tools([generate_sql_query, execute_sql_query, generate_natural_language_response])
        
        # First, generate the SQL query
        tool_call_prompt = f"""You need to answer a question about a music store database.
Follow these steps:
1. First, use the generate_sql_query tool to create a SQL query for this question: {user_query}
2. Then use the execute_sql_query tool to run the query
3. Finally, use the generate_natural_language_response tool to create a natural language answer

Start by generating the SQL query."""
        
        response = llm_with_tools.invoke([HumanMessage(content=tool_call_prompt)])
        return {"messages": [response]}
    
    # Handle tool responses and continue the workflow
    elif isinstance(last_message, ToolMessage):
        # After tools have been called, check if we need to call more tools
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
        llm_with_tools = llm.bind_tools([generate_sql_query, execute_sql_query, generate_natural_language_response])
        
        # Analyze the conversation to determine next steps
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # For AI messages with no tool calls, just pass through
    return {"messages": []}


def should_continue(state: State) -> str:
    """
    Determine whether to continue processing or end.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end
    return END


# Build the graph
def create_graph():
    """Create and compile the LangGraph workflow"""
    
    # Initialize the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("chatbot", chatbot_node)
    
    # Create tool node with all our tools
    tools = [generate_sql_query, execute_sql_query, generate_natural_language_response]
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges(
        "chatbot",
        should_continue,
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


