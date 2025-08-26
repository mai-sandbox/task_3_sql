import sqlite3
import requests
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
import os
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[BaseMessage]
    sql_query: str
    query_result: str
    final_answer: str


def setup_chinook_database():
    """Fetch Chinook database SQL and create in-memory SQLite database"""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    
    response = requests.get(url)
    response.raise_for_status()
    
    sql_script = response.text
    
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.executescript(sql_script)
    conn.commit()
    
    return conn


def get_schema_info(db: SQLDatabase) -> str:
    """Get detailed schema information for all tables"""
    table_info = db.get_table_info()
    return table_info


def generate_sql(state: AgentState) -> Dict:
    """Generate SQL query from natural language"""
    db = SQLDatabase.from_uri("sqlite:///chinook.db")
    schema_info = get_schema_info(db)
    
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert. Convert natural language questions to SQL queries for the Chinook database.
        
Database Schema:
{schema}

Important Instructions:
- Generate ONLY valid SQL queries that work with SQLite
- If the question is not related to the database or cannot be answered with SQL, respond with "NOT_RELEVANT"
- Be precise with table and column names
- Use proper JOIN statements when needed
- Ensure queries are safe and read-only (SELECT statements only)"""),
        ("human", "{question}")
    ])
    
    chain = prompt | model
    
    last_message = state["messages"][-1]
    question = last_message.content if isinstance(last_message, HumanMessage) else str(last_message)
    
    result = chain.invoke({
        "schema": schema_info,
        "question": question
    })
    
    sql_query = result.content.strip()
    
    if sql_query == "NOT_RELEVANT":
        return {
            "sql_query": "",
            "messages": state["messages"] + [AIMessage(content="I don't know the answer. This question cannot be answered using the Chinook database.")]
        }
    
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    
    return {"sql_query": sql_query, "messages": state["messages"]}


def execute_sql(state: AgentState) -> Dict:
    """Execute SQL query and return results"""
    if not state.get("sql_query"):
        return state
    
    db = SQLDatabase.from_uri("sqlite:///chinook.db")
    
    try:
        result = db.run(state["sql_query"])
        return {"query_result": str(result)}
    except Exception as e:
        return {"query_result": f"Error executing query: {str(e)}"}


def generate_response(state: AgentState) -> Dict:
    """Generate natural language response based on SQL results"""
    if not state.get("sql_query"):
        return state
    
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on SQL query results.
        
Instructions:
- Provide clear, natural language answers based on the query results
- Format the response in a user-friendly way
- If there's an error, explain it clearly
- Be concise but informative"""),
        ("human", """Question: {question}
SQL Query: {sql_query}
Query Result: {query_result}

Please provide a natural language answer based on the above information.""")
    ])
    
    chain = prompt | model
    
    last_message = state["messages"][-1]
    question = last_message.content if isinstance(last_message, HumanMessage) else str(last_message)
    
    response = chain.invoke({
        "question": question,
        "sql_query": state["sql_query"],
        "query_result": state.get("query_result", "No result")
    })
    
    return {
        "final_answer": response.content,
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }


def should_continue(state: AgentState) -> str:
    """Determine if we should continue processing"""
    if not state.get("sql_query"):
        return "end"
    return "continue"


def build_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("generate_response", generate_response)
    
    workflow.set_entry_point("generate_sql")
    
    workflow.add_conditional_edges(
        "generate_sql",
        should_continue,
        {
            "continue": "execute_sql",
            "end": END
        }
    )
    
    workflow.add_edge("execute_sql", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()


conn = setup_chinook_database()
conn.backup(sqlite3.connect("chinook.db"))
conn.close()

app = build_graph()


if __name__ == "__main__":
    print("Setting up Chinook database...")
    print("Database ready!")
    
    test_state = {
        "messages": [HumanMessage("How many customers are from Canada?")]
    }
    
    result = app.invoke(test_state)
    print("\nTest Query Result:")
    print(result.get("final_answer", result["messages"][-1].content))