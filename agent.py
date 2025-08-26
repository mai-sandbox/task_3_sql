import sqlite3
import requests
import os
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def create_chinook_db():
    """Create in-memory SQLite database from Chinook SQL file."""
    sql_url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(sql_url)
    response.raise_for_status()
    
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(response.text)
    return conn


db_conn = create_chinook_db()


def get_table_schema():
    """Get the schema of all tables in the Chinook database."""
    cursor = db_conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = []
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        column_info = []
        for col in columns:
            column_info.append(f"  - {col[1]} ({col[2]})")
        
        schema_info.append(f"Table: {table_name}\nColumns:\n" + "\n".join(column_info))
    
    return "\n\n".join(schema_info)


TABLE_SCHEMA = get_table_schema()


@tool
def execute_sql(query: str) -> str:
    """Execute SQL query against the Chinook database and return results."""
    try:
        cursor = db_conn.cursor()
        cursor.execute(query)
        
        if query.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
            if results:
                columns = [desc[0] for desc in cursor.description]
                result_str = f"Columns: {', '.join(columns)}\n"
                result_str += "\nResults:\n"
                for row in results[:50]:  # Limit to 50 rows for readability
                    result_str += str(row) + "\n"
                if len(results) > 50:
                    result_str += f"\n... and {len(results) - 50} more rows"
                return result_str
            else:
                return "Query executed successfully but returned no results."
        else:
            db_conn.commit()
            return f"Query executed successfully. Rows affected: {cursor.rowcount}"
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"


def create_agent():
    """Create the text-to-SQL agent."""
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    
    tools = [execute_sql]
    tool_node = ToolNode(tools)
    
    def generate_sql(state: AgentState):
        """Generate SQL query based on user's natural language request."""
        messages = state["messages"]
        
        system_prompt = f"""You are a SQL expert assistant for the Chinook music database. 
Your task is to:
1. Convert natural language requests into SQL queries
2. Execute the SQL queries
3. Generate a natural language response based on the results

Here is the database schema:
{TABLE_SCHEMA}

Important instructions:
- Only answer questions that can be answered using the Chinook database
- If a question is irrelevant or cannot be answered using SQL, respond with "I don't know the answer" and don't discuss anything else
- Always execute the SQL query you generate using the execute_sql tool
- Provide clear, concise natural language responses based on the query results
- Your sole purpose is to convert text requests to SQL and generate natural language responses"""
        
        response = llm.bind_tools(tools).invoke(
            [{"role": "system", "content": system_prompt}] + messages
        )
        
        return {"messages": [response]}
    
    def check_for_tools(state: AgentState):
        """Check if the last message has tool calls."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "respond"
    
    def generate_response(state: AgentState):
        """Generate final natural language response."""
        messages = state["messages"]
        
        system_prompt = """Based on the SQL query results, provide a clear and concise natural language answer to the user's question.
If the query failed or returned no relevant results, politely indicate that you cannot answer the question.
Remember: only answer questions related to the Chinook database."""
        
        response = llm.invoke(
            [{"role": "system", "content": system_prompt}] + messages
        )
        
        return {"messages": [response]}
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("tools", tool_node)
    workflow.add_node("respond", generate_response)
    
    workflow.set_entry_point("generate_sql")
    
    workflow.add_conditional_edges(
        "generate_sql",
        check_for_tools,
        {
            "tools": "tools",
            "respond": "respond"
        }
    )
    
    workflow.add_edge("tools", "respond")
    workflow.add_edge("respond", END)
    
    return workflow.compile()


app = create_agent()