import sqlite3
import requests
from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import re
import os
from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    messages: Annotated[List, add_messages]
    sql_query: str
    query_result: str
    database_schema: str


def fetch_and_setup_database():
    """Fetch Chinook database and set up in-memory SQLite"""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text
    
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    for statement in sql_script.split(';'):
        if statement.strip():
            try:
                cursor.execute(statement)
            except sqlite3.Error:
                continue
    
    conn.commit()
    return conn


def get_database_schema(conn):
    """Extract database schema information"""
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = []
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        schema_info.append(f"Table: {table_name}")
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            is_pk = "PRIMARY KEY" if col[5] else ""
            schema_info.append(f"  - {col_name} ({col_type}) {is_pk}")
        schema_info.append("")
    
    return "\n".join(schema_info)


conn = fetch_and_setup_database()
db_schema = get_database_schema(conn)

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def generate_sql(state: State) -> State:
    """Generate SQL query from natural language"""
    
    user_query = state["messages"][-1].content if state["messages"] else ""
    
    system_prompt = f"""You are a SQL expert. Convert natural language queries to SQL for a Chinook database.
    
Database Schema:
{db_schema}

Important:
- Generate ONLY valid SQL queries
- Use proper SQL syntax for SQLite
- Return ONLY the SQL query, nothing else
- If the query cannot be answered with the database, respond with "UNABLE_TO_ANSWER"
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Convert this to SQL: {user_query}")
    ]
    
    response = llm.invoke(messages)
    sql_query = response.content.strip()
    
    sql_query = re.sub(r'^```sql?\s*', '', sql_query)
    sql_query = re.sub(r'\s*```$', '', sql_query)
    
    state["sql_query"] = sql_query
    return state


def execute_sql(state: State) -> State:
    """Execute SQL query against the database"""
    
    sql_query = state.get("sql_query", "")
    
    if not sql_query or sql_query == "UNABLE_TO_ANSWER":
        state["query_result"] = "UNABLE_TO_ANSWER"
        return state
    
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        if sql_query.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            if results:
                result_str = f"Columns: {', '.join(columns)}\n"
                result_str += "\n".join([str(row) for row in results[:50]])
                if len(results) > 50:
                    result_str += f"\n... and {len(results) - 50} more rows"
            else:
                result_str = "No results found"
        else:
            conn.commit()
            result_str = f"Query executed successfully. Rows affected: {cursor.rowcount}"
        
        state["query_result"] = result_str
        
    except sqlite3.Error as e:
        state["query_result"] = f"SQL Error: {str(e)}"
    
    return state


def generate_response(state: State) -> State:
    """Generate natural language response based on query results"""
    
    user_query = state["messages"][-1].content if state["messages"] else ""
    sql_query = state.get("sql_query", "")
    query_result = state.get("query_result", "")
    
    if query_result == "UNABLE_TO_ANSWER" or sql_query == "UNABLE_TO_ANSWER":
        response_content = "I don't know the answer to that question. I can only answer questions about the Chinook music database."
    elif "SQL Error" in query_result:
        response_content = "I encountered an error while trying to answer your question. Please try rephrasing your query."
    else:
        system_prompt = """You are a helpful assistant that explains SQL query results in natural language.
Be concise and clear in your explanations.
If the results are empty, say so clearly."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
User Question: {user_query}
SQL Query Used: {sql_query}
Query Results: {query_result}

Please provide a natural language answer to the user's question based on these results.""")
        ]
        
        response = llm.invoke(messages)
        response_content = response.content
    
    state["messages"].append(AIMessage(content=response_content))
    return state


def should_continue(state: State) -> str:
    """Determine if we should continue or end"""
    if state.get("sql_query") == "UNABLE_TO_ANSWER":
        return "generate_response"
    return "execute_sql"


workflow = StateGraph(State)

workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("generate_response", generate_response)

workflow.set_entry_point("generate_sql")

workflow.add_conditional_edges(
    "generate_sql",
    should_continue,
    {
        "execute_sql": "execute_sql",
        "generate_response": "generate_response"
    }
)

workflow.add_edge("execute_sql", "generate_response")
workflow.add_edge("generate_response", END)

app = workflow.compile()