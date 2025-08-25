"""
LangGraph Text-to-SQL Agent for Chinook Database

This agent converts natural language queries to SQL, executes them against 
the Chinook SQLite database, and returns natural language responses.
"""

import os
import sqlite3
import requests
from typing import TypedDict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from sqlalchemy import create_engine, text, inspect
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database URL for Chinook SQLite
CHINOOK_SQL_URL = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"

class AgentState(TypedDict):
    """State for the SQL agent"""
    messages: List[BaseMessage]
    user_query: Optional[str]
    sql_query: Optional[str]
    sql_result: Optional[Any]
    error: Optional[str]
    schema_info: Optional[str]

class ChinookSQLAgent:
    """Text-to-SQL Agent for Chinook Database"""
    
    def __init__(self):
        self.engine = None
        self.schema_info = ""
        self.llm = None  # Initialize LLM lazily
        self._setup_database()
        self._extract_schema()
    
    def _get_llm(self):
        """Get or initialize the LLM based on available API keys"""
        if self.llm is None:
            if os.getenv("OPENAI_API_KEY"):
                self.llm = ChatOpenAI(model="gpt-4", temperature=0)
            elif os.getenv("ANTHROPIC_API_KEY"):
                self.llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
            else:
                # Default to OpenAI - will need API key at runtime
                self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        return self.llm
    
    def _setup_database(self):
        """Fetch Chinook SQL and create in-memory SQLite database"""
        try:
            logger.info("Fetching Chinook database SQL...")
            response = requests.get(CHINOOK_SQL_URL, timeout=30)
            response.raise_for_status()
            
            # Create in-memory SQLite database
            self.engine = create_engine("sqlite:///:memory:")
            
            # Execute the SQL to populate the database
            with self.engine.connect() as conn:
                # Split SQL into individual statements and execute
                sql_statements = response.text.split(';')
                for statement in sql_statements:
                    statement = statement.strip()
                    if statement and not statement.startswith('/*') and not statement.startswith('--'):
                        try:
                            conn.execute(text(statement))
                        except Exception as e:
                            # Skip problematic statements (like comments)
                            continue
                conn.commit()
            
            logger.info("Chinook database successfully loaded into memory")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    def _extract_schema(self):
        """Extract and format database schema information"""
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            schema_parts = ["# Chinook Database Schema\n"]
            
            for table in tables:
                columns = inspector.get_columns(table)
                foreign_keys = inspector.get_foreign_keys(table)
                
                schema_parts.append(f"## Table: {table}")
                schema_parts.append("Columns:")
                
                for col in columns:
                    col_info = f"- {col['name']}: {col['type']}"
                    if not col.get('nullable', True):
                        col_info += " (NOT NULL)"
                    if col.get('primary_key', False):
                        col_info += " (PRIMARY KEY)"
                    schema_parts.append(col_info)
                
                if foreign_keys:
                    schema_parts.append("Foreign Keys:")
                    for fk in foreign_keys:
                        fk_info = f"- {fk['constrained_columns'][0]} -> {fk['referred_table']}.{fk['referred_columns'][0]}"
                        schema_parts.append(fk_info)
                
                schema_parts.append("")  # Empty line between tables
            
            self.schema_info = "\n".join(schema_parts)
            logger.info("Database schema extracted successfully")
            
        except Exception as e:
            logger.error(f"Error extracting schema: {e}")
            self.schema_info = "Schema information unavailable"

def analyze_query(state: AgentState) -> AgentState:
    """Analyze the user query to determine if it's relevant to the database"""
    messages = state["messages"]
    if not messages:
        state["error"] = "No messages provided"
        return state
    
    # Get the last human message
    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    if not user_message:
        state["error"] = "No user query found"
        return state
    
    state["user_query"] = user_message
    return state

def generate_sql(state: AgentState) -> AgentState:
    """Generate SQL query from natural language"""
    if state.get("error"):
        return state
    
    user_query = state["user_query"]
    schema_info = state.get("schema_info", "")
    
    # Create the SQL generation prompt
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert working with a Chinook music database. 
Your task is to convert natural language questions into valid SQLite queries.

{schema_info}

Rules:
1. Only generate SQL queries that can be answered using the Chinook database tables shown above
2. If the question is not related to music, albums, artists, customers, invoices, or employees, respond with "IRRELEVANT"
3. Generate only the SQL query, no explanations
4. Use proper SQLite syntax
5. Be precise with table and column names
6. Use appropriate JOINs when needed
7. Limit results to reasonable numbers (e.g., TOP 10) unless specifically asked otherwise

Examples:
- "Show me all albums by AC/DC" -> SELECT Album.Title FROM Album JOIN Artist ON Album.ArtistId = Artist.ArtistId WHERE Artist.Name = 'AC/DC'
- "What's the weather like?" -> IRRELEVANT
"""),
        ("human", "{query}")
    ])
    
    try:
        # Initialize LLM
        if os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(model="gpt-4", temperature=0)
        elif os.getenv("ANTHROPIC_API_KEY"):
            llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
        else:
            llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        chain = sql_prompt | llm
        response = chain.invoke({
            "schema_info": schema_info,
            "query": user_query
        })
        
        sql_query = response.content.strip()
        
        if sql_query == "IRRELEVANT":
            state["error"] = "IRRELEVANT_QUERY"
        else:
            state["sql_query"] = sql_query
        
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        state["error"] = f"SQL generation failed: {str(e)}"
    
    return state

def execute_sql(state: AgentState) -> AgentState:
    """Execute the generated SQL query"""
    if state.get("error"):
        return state
    
    sql_query = state.get("sql_query")
    if not sql_query:
        state["error"] = "No SQL query to execute"
        return state
    
    try:
        # Get the engine from the global agent instance
        engine = create_engine("sqlite:///:memory:")
        
        # Re-setup database for this execution (in a real scenario, we'd maintain the connection)
        response = requests.get(CHINOOK_SQL_URL, timeout=30)
        response.raise_for_status()
        
        with engine.connect() as conn:
            # Re-populate database
            sql_statements = response.text.split(';')
            for statement in sql_statements:
                statement = statement.strip()
                if statement and not statement.startswith('/*') and not statement.startswith('--'):
                    try:
                        conn.execute(text(statement))
                    except Exception:
                        continue
            conn.commit()
            
            # Execute the user's query
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            columns = result.keys()
            
            # Format results
            if rows:
                formatted_results = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    formatted_results.append(row_dict)
                state["sql_result"] = formatted_results
            else:
                state["sql_result"] = []
        
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        state["error"] = f"SQL execution failed: {str(e)}"
    
    return state

def generate_response(state: AgentState) -> AgentState:
    """Generate natural language response based on SQL results"""
    if state.get("error"):
        if state["error"] == "IRRELEVANT_QUERY":
            response = "I don't know the answer to that question. I can only help with questions about the music database, including artists, albums, tracks, customers, and sales information."
        else:
            response = "I don't know the answer to that question."
    else:
        user_query = state["user_query"]
        sql_result = state.get("sql_result", [])
        
        # Create response generation prompt
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that explains database query results in natural language.
Given a user's question and the corresponding SQL query results, provide a clear, concise answer.

Rules:
1. Be conversational and helpful
2. Present the data in an easy-to-read format
3. If there are no results, say so politely
4. Don't mention SQL or technical details
5. Focus on answering the user's original question
"""),
            ("human", """User asked: {query}

Query results: {results}

Please provide a natural language response to the user's question based on these results.""")
        ])
        
        try:
            llm = ChatOpenAI(model="gpt-4", temperature=0) if os.getenv("OPENAI_API_KEY") else ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
            
            chain = response_prompt | llm
            ai_response = chain.invoke({
                "query": user_query,
                "results": sql_result
            })
            
            response = ai_response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = "I found some results but had trouble formatting the response."
    
    # Add the response to messages
    messages = state["messages"].copy()
    messages.append(AIMessage(content=response))
    state["messages"] = messages
    
    return state

# Create the agent workflow
def create_agent():
    """Create and compile the LangGraph agent"""
    
    # Initialize the database and schema
    agent = ChinookSQLAgent()
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("generate_response", generate_response)
    
    # Set entry point
    workflow.set_entry_point("analyze_query")
    
    # Add edges
    workflow.add_edge("analyze_query", "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Compile the graph
    return workflow.compile()

# Create and export the compiled graph
app = create_agent()

# Add schema info to the initial state
def _add_schema_to_state(state: AgentState) -> AgentState:
    """Add schema information to state"""
    if not state.get("schema_info"):
        try:
            agent = ChinookSQLAgent()
            state["schema_info"] = agent.schema_info
        except Exception as e:
            logger.error(f"Error adding schema to state: {e}")
            state["schema_info"] = "Schema unavailable"
    return state

# Wrap the app to ensure schema is available
original_invoke = app.invoke

def enhanced_invoke(state):
    """Enhanced invoke that ensures schema is available"""
    state = _add_schema_to_state(state)
    return original_invoke(state)

app.invoke = enhanced_invoke


