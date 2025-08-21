"""
LangGraph Text-to-SQL Agent for Chinook Database

This agent converts natural language queries to SQL, executes them against 
the Chinook SQLite database, and returns natural language responses.
"""

import sqlite3
import requests
from typing import Dict, Any, List, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
import os


class AgentState(TypedDict):
    """State for the text-to-SQL agent"""
    messages: List[Any]
    query: str
    sql: str
    sql_result: Any
    schema_info: str
    error: str
    response: str


class ChinookDatabase:
    """Manages the Chinook SQLite database"""
    
    def __init__(self):
        self.connection = None
        self.schema_info = ""
    
    def initialize_database(self) -> bool:
        """Fetch Chinook SQL and create in-memory database"""
        try:
            # Fetch the Chinook SQL file
            url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Create in-memory SQLite database
            self.connection = sqlite3.connect(":memory:")
            cursor = self.connection.cursor()
            
            # Execute the SQL to create and populate the database
            cursor.executescript(response.text)
            self.connection.commit()
            
            # Extract schema information
            self._extract_schema_info()
            
            return True
            
        except Exception as e:
            print(f"Error initializing database: {e}")
            return False
    
    def _extract_schema_info(self):
        """Extract detailed schema information from the database"""
        cursor = self.connection.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cursor.fetchall()
        
        schema_parts = []
        schema_parts.append("=== CHINOOK DATABASE SCHEMA ===\n")
        
        for (table_name,) in tables:
            schema_parts.append(f"Table: {table_name}")
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, is_pk = col
                pk_indicator = " (PRIMARY KEY)" if is_pk else ""
                null_indicator = " NOT NULL" if not_null else ""
                schema_parts.append(f"  - {col_name}: {col_type}{pk_indicator}{null_indicator}")
            
            # Get foreign key information
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = cursor.fetchall()
            
            for fk in foreign_keys:
                fk_id, seq, ref_table, from_col, to_col, on_update, on_delete, match = fk
                schema_parts.append(f"  - FOREIGN KEY: {from_col} -> {ref_table}({to_col})")
            
            schema_parts.append("")  # Empty line between tables
        
        # Add some sample data context
        schema_parts.append("=== SAMPLE DATA CONTEXT ===")
        schema_parts.append("This is a music store database containing:")
        schema_parts.append("- Artists and their albums")
        schema_parts.append("- Tracks with genres and media types")
        schema_parts.append("- Customers and their purchase invoices")
        schema_parts.append("- Employees and their customer relationships")
        schema_parts.append("- Playlists and track associations")
        
        self.schema_info = "\n".join(schema_parts)
    
    def execute_query(self, sql: str) -> tuple:
        """Execute SQL query and return results"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            
            # For SELECT queries, fetch results
            if sql.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                # Get column names
                column_names = [description[0] for description in cursor.description]
                return True, {"columns": column_names, "rows": results}
            else:
                # For other queries, return affected rows
                return True, {"affected_rows": cursor.rowcount}
                
        except Exception as e:
            return False, str(e)


# Initialize the database
db = ChinookDatabase()
if not db.initialize_database():
    raise RuntimeError("Failed to initialize Chinook database")


def extract_query_node(state: AgentState) -> AgentState:
    """Extract the user's query from the messages"""
    messages = state["messages"]
    if messages and isinstance(messages[-1], HumanMessage):
        state["query"] = messages[-1].content
    else:
        state["query"] = ""
        state["error"] = "No valid query found in messages"
    return state


def generate_sql_node(state: AgentState) -> AgentState:
    """Generate SQL from natural language query using LLM"""
    if state.get("error"):
        return state
    
    query = state["query"]
    schema_info = db.schema_info
    
    # Initialize the LLM
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    prompt = f"""You are a SQL expert. Convert the following natural language query to SQL for the Chinook database.

{schema_info}

Rules:
1. Only generate SELECT statements for data retrieval
2. Use proper table joins when needed
3. Return only the SQL query, no explanations
4. If the query is not related to music, artists, albums, customers, or sales data, return "IRRELEVANT_QUERY"
5. Use LIMIT to restrict results to reasonable numbers (e.g., LIMIT 10 for lists)

User Query: {query}

SQL Query:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        sql = response.content.strip()
        
        # Check if query is irrelevant
        if sql == "IRRELEVANT_QUERY":
            state["error"] = "IRRELEVANT_QUERY"
        else:
            # Clean up the SQL (remove markdown formatting if present)
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.endswith("```"):
                sql = sql[:-3]
            state["sql"] = sql.strip()
            
    except Exception as e:
        state["error"] = f"Error generating SQL: {str(e)}"
    
    return state


def execute_sql_node(state: AgentState) -> AgentState:
    """Execute the generated SQL against the database"""
    if state.get("error"):
        return state
    
    sql = state.get("sql", "")
    if not sql:
        state["error"] = "No SQL query to execute"
        return state
    
    success, result = db.execute_query(sql)
    
    if success:
        state["sql_result"] = result
    else:
        state["error"] = f"SQL execution error: {result}"
    
    return state


def generate_response_node(state: AgentState) -> AgentState:
    """Generate natural language response from SQL results"""
    if state.get("error"):
        # Handle irrelevant queries
        if state["error"] == "IRRELEVANT_QUERY":
            state["response"] = "I don't know the answer to that question. I can only help with queries related to the music store database, such as information about artists, albums, tracks, customers, and sales."
        else:
            state["response"] = "I don't know the answer to that question."
        return state
    
    query = state["query"]
    sql = state["sql"]
    sql_result = state["sql_result"]
    
    # Initialize the LLM
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Format the SQL results for the prompt
    if isinstance(sql_result, dict) and "columns" in sql_result:
        columns = sql_result["columns"]
        rows = sql_result["rows"]
        
        if not rows:
            result_text = "No results found."
        else:
            # Format as a simple table
            result_lines = [" | ".join(columns)]
            result_lines.append("-" * len(result_lines[0]))
            for row in rows[:10]:  # Limit to first 10 rows for display
                result_lines.append(" | ".join(str(cell) for cell in row))
            result_text = "\n".join(result_lines)
    else:
        result_text = str(sql_result)
    
    prompt = f"""Convert the following SQL query results into a natural, conversational response to the user's question.

User's Question: {query}

SQL Query Used: {sql}

Query Results:
{result_text}

Instructions:
1. Provide a clear, natural language answer
2. Include specific details from the results
3. If there are many results, summarize appropriately
4. Be conversational and helpful
5. Don't mention the SQL query or technical details

Response:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        state["response"] = response.content.strip()
    except Exception as e:
        state["response"] = "I found some results but had trouble formatting the response."
    
    return state


def create_final_message_node(state: AgentState) -> AgentState:
    """Create the final AI message with the response"""
    response = state.get("response", "I don't know the answer to that question.")
    
    # Add the AI response to messages
    messages = state["messages"].copy()
    messages.append(AIMessage(content=response))
    state["messages"] = messages
    
    return state


# Create the LangGraph workflow
def create_workflow():
    """Create and return the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract_query", extract_query_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("create_final_message", create_final_message_node)
    
    # Define the flow
    workflow.set_entry_point("extract_query")
    workflow.add_edge("extract_query", "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "generate_response")
    workflow.add_edge("generate_response", "create_final_message")
    workflow.add_edge("create_final_message", END)
    
    return workflow


# Create and compile the graph
graph = create_workflow()
app = graph.compile()
