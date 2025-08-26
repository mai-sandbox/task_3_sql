"""
LangGraph Text-to-SQL Agent for Chinook Database

This agent converts natural language queries to SQL, executes them against the Chinook database,
and returns natural language responses.
"""

import sqlite3
import requests
from typing import TypedDict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os
import re


class AgentState(TypedDict):
    """State schema for the SQL agent."""
    messages: List[BaseMessage]
    sql_query: Optional[str]
    sql_result: Optional[Any]
    error: Optional[str]


class ChinookDatabase:
    """Manages the Chinook SQLite database."""
    
    def __init__(self):
        self.connection = None
        self.schema_info = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Download and initialize the Chinook database in memory."""
        try:
            # Download the SQL file
            url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
            response = requests.get(url)
            response.raise_for_status()
            
            # Create in-memory SQLite database
            self.connection = sqlite3.connect(":memory:")
            cursor = self.connection.cursor()
            
            # Execute the SQL script
            cursor.executescript(response.text)
            self.connection.commit()
            
            # Get schema information
            self._extract_schema_info()
            
        except Exception as e:
            raise Exception(f"Failed to initialize Chinook database: {str(e)}")
    
    def _extract_schema_info(self):
        """Extract detailed schema information for the LLM."""
        cursor = self.connection.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_info = {}
        for (table_name,) in tables:
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Get foreign key information
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = cursor.fetchall()
            
            schema_info[table_name] = {
                'columns': columns,
                'foreign_keys': foreign_keys
            }
        
        self.schema_info = schema_info
    
    def get_schema_description(self) -> str:
        """Generate a comprehensive schema description for the LLM."""
        description = "CHINOOK DATABASE SCHEMA:\n\n"
        
        for table_name, info in self.schema_info.items():
            description += f"Table: {table_name}\n"
            description += "Columns:\n"
            
            for col in info['columns']:
                col_id, name, data_type, not_null, default_val, pk = col
                pk_indicator = " (PRIMARY KEY)" if pk else ""
                not_null_indicator = " NOT NULL" if not_null else ""
                description += f"  - {name}: {data_type}{not_null_indicator}{pk_indicator}\n"
            
            if info['foreign_keys']:
                description += "Foreign Keys:\n"
                for fk in info['foreign_keys']:
                    fk_id, seq, table, from_col, to_col, on_update, on_delete, match = fk
                    description += f"  - {from_col} -> {table}.{to_col}\n"
            
            description += "\n"
        
        # Add some sample data context
        description += "IMPORTANT RELATIONSHIPS:\n"
        description += "- Artist -> Album (ArtistId)\n"
        description += "- Album -> Track (AlbumId)\n"
        description += "- Track -> InvoiceLine (TrackId)\n"
        description += "- Invoice -> InvoiceLine (InvoiceId)\n"
        description += "- Customer -> Invoice (CustomerId)\n"
        description += "- Employee -> Customer (SupportRepId)\n"
        description += "- Genre -> Track (GenreId)\n"
        description += "- MediaType -> Track (MediaTypeId)\n"
        description += "- Playlist -> PlaylistTrack (PlaylistId)\n"
        description += "- Track -> PlaylistTrack (TrackId)\n\n"
        
        return description
    
    def execute_query(self, query: str) -> Any:
        """Execute a SQL query and return results."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # For SELECT queries, fetch results
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                # Get column names
                column_names = [description[0] for description in cursor.description]
                return {'columns': column_names, 'rows': results}
            else:
                # For other queries, return affected rows
                return {'affected_rows': cursor.rowcount}
                
        except Exception as e:
            raise Exception(f"SQL execution error: {str(e)}")


# Initialize the database
db = ChinookDatabase()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


def is_relevant_query(user_input: str) -> bool:
    """Check if the user query is relevant to the music database."""
    music_keywords = [
        'artist', 'album', 'track', 'song', 'music', 'genre', 'playlist',
        'customer', 'invoice', 'sales', 'purchase', 'employee', 'media',
        'composer', 'band', 'singer', 'musician', 'record', 'cd', 'mp3',
        'rock', 'jazz', 'pop', 'classical', 'metal', 'blues', 'country'
    ]
    
    # Check for database-related queries
    db_keywords = ['how many', 'count', 'list', 'show', 'find', 'search', 'top', 'best', 'most', 'least']
    
    user_lower = user_input.lower()
    
    # Check if query contains music-related or database query keywords
    has_music_keyword = any(keyword in user_lower for keyword in music_keywords)
    has_db_keyword = any(keyword in user_lower for keyword in db_keywords)
    
    return has_music_keyword or has_db_keyword


def generate_sql_node(state: AgentState) -> AgentState:
    """Generate SQL query from natural language input."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the query is relevant
    if isinstance(last_message, HumanMessage):
        if not is_relevant_query(last_message.content):
            state["error"] = "irrelevant_query"
            return state
    
    # Create the SQL generation prompt
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a SQL expert. Convert the user's natural language query to a valid SQLite query for the Chinook music database.

{db.get_schema_description()}

IMPORTANT RULES:
1. Only generate SELECT queries - no INSERT, UPDATE, DELETE, or DDL statements
2. Use proper SQLite syntax
3. Include appropriate JOINs when querying multiple tables
4. Use LIMIT clause for "top N" queries
5. Be case-insensitive in WHERE clauses using LOWER() function
6. Return only the SQL query, no explanations or markdown formatting
7. If the query cannot be answered with the available data, return "CANNOT_ANSWER"

Examples:
- "How many artists are there?" -> SELECT COUNT(*) FROM Artist;
- "Top 5 albums by sales" -> SELECT a.Title, SUM(il.Quantity) as sales FROM Album a JOIN Track t ON a.AlbumId = t.AlbumId JOIN InvoiceLine il ON t.TrackId = il.TrackId GROUP BY a.AlbumId, a.Title ORDER BY sales DESC LIMIT 5;
"""),
        ("human", "{query}")
    ])
    
    try:
        response = llm.invoke(sql_prompt.format_messages(query=last_message.content))
        sql_query = response.content.strip()
        
        if sql_query == "CANNOT_ANSWER":
            state["error"] = "cannot_answer"
        else:
            state["sql_query"] = sql_query
            
    except Exception as e:
        state["error"] = f"sql_generation_error: {str(e)}"
    
    return state


def execute_sql_node(state: AgentState) -> AgentState:
    """Execute the generated SQL query."""
    if state.get("error") or not state.get("sql_query"):
        return state
    
    try:
        result = db.execute_query(state["sql_query"])
        state["sql_result"] = result
    except Exception as e:
        state["error"] = f"sql_execution_error: {str(e)}"
    
    return state


def generate_response_node(state: AgentState) -> AgentState:
    """Generate natural language response from SQL results."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Handle error cases
    if state.get("error"):
        error = state["error"]
        if error == "irrelevant_query" or error == "cannot_answer":
            response = "I don't know the answer to that question."
        else:
            response = "I don't know the answer to that question."
        
        messages.append(AIMessage(content=response))
        return state
    
    # Generate response from SQL results
    sql_result = state.get("sql_result")
    sql_query = state.get("sql_query")
    
    if not sql_result:
        messages.append(AIMessage(content="I don't know the answer to that question."))
        return state
    
    # Create response generation prompt
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that converts SQL query results into natural language responses.

RULES:
1. Provide clear, concise answers based on the SQL results
2. Use natural language that's easy to understand
3. Include specific numbers and details from the results
4. If there are no results, say "No results found"
5. Format lists and tables in a readable way
6. Don't mention SQL or technical details in your response
"""),
        ("human", """
Original question: {question}
SQL query executed: {sql_query}
Results: {results}

Please provide a natural language response to the original question based on these results.
""")
    ])
    
    try:
        # Format results for the prompt
        if 'columns' in sql_result and 'rows' in sql_result:
            if not sql_result['rows']:
                formatted_results = "No results found"
            else:
                # Format as a readable table
                columns = sql_result['columns']
                rows = sql_result['rows']
                formatted_results = f"Columns: {', '.join(columns)}\n"
                formatted_results += "Data:\n"
                for row in rows[:10]:  # Limit to first 10 rows
                    formatted_results += f"  {', '.join(str(item) for item in row)}\n"
                if len(rows) > 10:
                    formatted_results += f"  ... and {len(rows) - 10} more rows"
        else:
            formatted_results = str(sql_result)
        
        response = llm.invoke(response_prompt.format_messages(
            question=last_message.content,
            sql_query=sql_query,
            results=formatted_results
        ))
        
        messages.append(AIMessage(content=response.content))
        
    except Exception as e:
        messages.append(AIMessage(content="I don't know the answer to that question."))
    
    return state


def should_continue(state: AgentState) -> str:
    """Determine the next step in the workflow."""
    if state.get("error"):
        return "generate_response"
    elif state.get("sql_query") and not state.get("sql_result"):
        return "execute_sql"
    elif state.get("sql_result"):
        return "generate_response"
    else:
        return "generate_sql"


# Create the StateGraph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("generate_response", generate_response_node)

# Set entry point
workflow.set_entry_point("generate_sql")

# Add edges
workflow.add_conditional_edges(
    "generate_sql",
    should_continue,
    {
        "execute_sql": "execute_sql",
        "generate_response": "generate_response"
    }
)

workflow.add_conditional_edges(
    "execute_sql",
    should_continue,
    {
        "generate_response": "generate_response"
    }
)

workflow.add_edge("generate_response", END)

# Compile the graph
app = workflow.compile()
