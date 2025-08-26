import sqlite3
import requests
from typing import TypedDict, Literal, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[BaseMessage]
    sql_query: str
    query_result: str
    error: str
    schema: str


def fetch_and_setup_database():
    """Fetch Chinook database and create in-memory SQLite database"""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text
    
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    cursor.executescript(sql_script)
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
        
        column_info = []
        for col in columns:
            column_info.append(f"  - {col[1]} ({col[2]})")
        
        schema_info.append(f"Table: {table_name}")
        schema_info.extend(column_info)
    
    return "\n".join(schema_info)


db_conn = fetch_and_setup_database()
db_schema = get_database_schema(db_conn)


def generate_sql(state: AgentState) -> AgentState:
    """Generate SQL query from natural language"""
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    system_prompt = f"""You are a SQL expert. Convert natural language questions into SQL queries for the Chinook database.
    
Database Schema:
{db_schema}

Important rules:
1. Only generate SELECT queries (no INSERT, UPDATE, DELETE)
2. Return ONLY the SQL query, no explanations
3. If the question is unrelated to the database or cannot be answered, respond with "INVALID_QUERY"
4. Be precise with table and column names based on the schema provided"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    last_message = state["messages"][-1].content
    
    chain = prompt | llm
    response = chain.invoke({"question": last_message})
    
    sql_query = response.content.strip()
    
    return {
        **state,
        "sql_query": sql_query,
        "schema": db_schema
    }


def execute_sql(state: AgentState) -> AgentState:
    """Execute SQL query against the database"""
    sql_query = state["sql_query"]
    
    if sql_query == "INVALID_QUERY":
        return {
            **state,
            "query_result": "",
            "error": "Cannot answer this question using the database"
        }
    
    try:
        cursor = db_conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            formatted_results = []
            formatted_results.append(columns)
            formatted_results.extend(results)
            query_result = str(formatted_results)
        else:
            query_result = str(results)
        
        return {
            **state,
            "query_result": query_result,
            "error": ""
        }
    except Exception as e:
        return {
            **state,
            "query_result": "",
            "error": str(e)
        }


def generate_response(state: AgentState) -> AgentState:
    """Generate natural language response based on SQL results"""
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    if state.get("error"):
        if "Cannot answer this question" in state["error"]:
            response_text = "I don't know the answer to that question. I can only help with queries related to the Chinook music database."
        else:
            response_text = f"I encountered an error while processing your query: {state['error']}"
    else:
        system_prompt = """You are a helpful assistant that explains SQL query results in natural language.
        Based on the SQL query and its results, provide a clear, concise answer to the user's question.
        Be specific and include relevant details from the results."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """Original question: {question}
            
SQL Query executed: {sql_query}

Query Results: {results}

Please provide a natural language answer based on these results.""")
        ])
        
        chain = prompt | llm
        response = chain.invoke({
            "question": state["messages"][-1].content,
            "sql_query": state["sql_query"],
            "results": state["query_result"]
        })
        
        response_text = response.content
    
    state["messages"].append(AIMessage(content=response_text))
    return state


def should_continue(state: AgentState) -> Literal["execute", "respond"]:
    """Determine if we should execute SQL or skip to response"""
    if state["sql_query"] == "INVALID_QUERY":
        return "respond"
    return "execute"


workflow = StateGraph(AgentState)

workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("generate_response", generate_response)

workflow.set_entry_point("generate_sql")

workflow.add_conditional_edges(
    "generate_sql",
    should_continue,
    {
        "execute": "execute_sql",
        "respond": "generate_response"
    }
)

workflow.add_edge("execute_sql", "generate_response")
workflow.add_edge("generate_response", END)

app = workflow.compile()