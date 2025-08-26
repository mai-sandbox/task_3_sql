"""
LangGraph Text-to-SQL Agent for Chinook Database

This agent converts natural language queries to SQL, executes them against
the Chinook SQLite database, and returns natural language responses.
"""

import sqlite3
import requests
from typing import TypedDict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    """State schema for the text-to-SQL agent."""

    messages: List[BaseMessage]
    sql_query: Optional[str]
    sql_results: Optional[List[Any]]
    database_connection: Optional[sqlite3.Connection]
    schema_info: Optional[str]
    error: Optional[str]


def initialize_database(state: AgentState) -> AgentState:
    """
    Initialize the Chinook SQLite database in memory by fetching the SQL script
    from the provided URL and executing it.
    """
    try:
        # Fetch the Chinook database SQL script
        url = (
            "https://raw.githubusercontent.com/lerocha/chinook-database/"
            "master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
        )
        response = requests.get(url)
        response.raise_for_status()

        # Create in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # Execute the SQL script to create and populate the database
        cursor.executescript(response.text)
        conn.commit()

        state["database_connection"] = conn
        return state

    except Exception as e:
        state["error"] = f"Failed to initialize database: {str(e)}"
        return state


def extract_schema_info(state: AgentState) -> AgentState:
    """
    Extract detailed schema information from the database to provide context
    for SQL generation.
    """
    if state.get("error") or not state.get("database_connection"):
        return state

    try:
        conn = state["database_connection"]
        cursor = conn.cursor()

        # Get all table names
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = cursor.fetchall()

        schema_info = "Database Schema Information:\n\n"

        for (table_name,) in tables:
            schema_info += f"Table: {table_name}\n"

            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            for col in columns:
                col_name = col[1]
                col_type = col[2]
                not_null = "NOT NULL" if col[3] else ""
                pk = "PRIMARY KEY" if col[5] else ""
                schema_info += (
                    f"  - {col_name} ({col_type}) {not_null} {pk}\n".strip() + "\n"
                )

            # Get foreign key information
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            fks = cursor.fetchall()

            for fk in fks:
                from_col = fk[3]
                to_table = fk[2]
                to_col = fk[4]
                schema_info += f"  - FOREIGN KEY: {from_col} -> {to_table}({to_col})\n"

            schema_info += "\n"

        state["schema_info"] = schema_info
        return state

    except Exception as e:
        state["error"] = f"Failed to extract schema: {str(e)}"
        return state


def generate_sql(state: AgentState) -> AgentState:
    """
    Generate SQL query from natural language using Anthropic Claude with schema context.
    """
    if state.get("error") or not state.get("schema_info"):
        return state

    try:
        # Get the latest human message
        human_messages = [
            msg for msg in state["messages"] if isinstance(msg, HumanMessage)
        ]
        if not human_messages:
            state["error"] = "No human message found"
            return state

        user_query = human_messages[-1].content

        # Initialize Claude
        llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

        # Create prompt with schema context
        prompt = f"""You are a SQL expert working with a Chinook music database.

{state["schema_info"]}

Convert the following natural language query to SQL. Follow these rules:
1. Only generate SELECT queries - no INSERT, UPDATE, DELETE, or DDL statements
2. Use proper SQL syntax for SQLite
3. If the query cannot be answered with the available tables, respond with exactly:
   "IRRELEVANT_QUERY"
4. Return only the SQL query, no explanations or formatting
5. Use appropriate JOINs when needed to get complete information
6. Limit results to reasonable numbers (e.g., TOP 10 or LIMIT 10)

User Query: {user_query}

SQL Query:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        sql_query = response.content.strip()

        # Check if query is irrelevant
        if "IRRELEVANT_QUERY" in sql_query:
            state["sql_query"] = None
            state["error"] = "IRRELEVANT_QUERY"
        else:
            state["sql_query"] = sql_query

        return state

    except Exception as e:
        state["error"] = f"Failed to generate SQL: {str(e)}"
        return state


def execute_sql(state: AgentState) -> AgentState:
    """
    Execute the generated SQL query against the database.
    """
    if (
        state.get("error")
        or not state.get("sql_query")
        or not state.get("database_connection")
    ):
        return state

    try:
        conn = state["database_connection"]
        cursor = conn.cursor()

        # Execute the SQL query
        cursor.execute(state["sql_query"])
        results = cursor.fetchall()

        # Get column names
        column_names = (
            [description[0] for description in cursor.description]
            if cursor.description
            else []
        )

        # Format results as list of dictionaries
        formatted_results = []
        for row in results:
            formatted_results.append(dict(zip(column_names, row)))

        state["sql_results"] = formatted_results
        return state

    except Exception as e:
        state["error"] = f"Failed to execute SQL: {str(e)}"
        return state


def generate_response(state: AgentState) -> AgentState:
    """
    Generate natural language response based on SQL results or handle errors.
    """
    try:
        # Handle irrelevant queries
        if state.get("error") == "IRRELEVANT_QUERY":
            response = "I don't know the answer to that question. I can only help with queries related to the music database."
            state["messages"].append(AIMessage(content=response))
            return state

        # Handle other errors
        if state.get("error"):
            response = "I don't know the answer to that question."
            state["messages"].append(AIMessage(content=response))
            return state

        # Handle empty results
        if not state.get("sql_results"):
            response = "I found no results for your query."
            state["messages"].append(AIMessage(content=response))
            return state

        # Get the original user query
        human_messages = [
            msg for msg in state["messages"] if isinstance(msg, HumanMessage)
        ]
        user_query = human_messages[-1].content if human_messages else "the query"

        # Initialize Claude for response generation
        llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.3)

        # Format results for the prompt
        results_text = ""
        for i, result in enumerate(
            state["sql_results"][:10]
        ):  # Limit to first 10 results
            results_text += f"Result {i+1}: {result}\n"

        if len(state["sql_results"]) > 10:
            results_text += f"... and {len(state['sql_results']) - 10} more results\n"

        # Create prompt for natural language response
        prompt = f"""Based on the following SQL query results, provide a clear and natural language answer to the user's question.

User's Question: {user_query}

SQL Query Used: {state["sql_query"]}

Query Results:
{results_text}

Instructions:
1. Provide a direct, helpful answer in natural language
2. Include specific details from the results when relevant
3. If there are many results, summarize the key findings
4. Be conversational and informative
5. Don't mention SQL or technical details unless necessary

Natural Language Response:"""

        response = llm.invoke([HumanMessage(content=prompt)])

        state["messages"].append(AIMessage(content=response.content))
        return state

    except Exception as e:
        response = "I don't know the answer to that question."
        state["messages"].append(AIMessage(content=response))
        return state


def should_continue(state: AgentState) -> str:
    """Determine if the workflow should continue or end."""
    return END


# Create the LangGraph workflow
def create_workflow():
    """Create and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("initialize_db", initialize_database)
    workflow.add_node("extract_schema", extract_schema_info)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("generate_response", generate_response)

    # Define the workflow edges
    workflow.set_entry_point("initialize_db")
    workflow.add_edge("initialize_db", "extract_schema")
    workflow.add_edge("extract_schema", "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()


# Export the compiled graph as 'app' following LangGraph pattern
app = create_workflow()





