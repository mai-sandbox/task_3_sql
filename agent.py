"""
LangGraph Text-to-SQL Agent for Chinook Database

This agent converts natural language queries to SQL, executes them against 
the Chinook SQLite database, and returns natural language responses.
"""

import sqlite3
import requests
from typing import TypedDict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json
import re


class AgentState(TypedDict):
    """State for the text-to-SQL agent"""
    messages: List[BaseMessage]
    database_initialized: bool
    schema_info: Optional[str]
    user_query: Optional[str]
    generated_sql: Optional[str]
    sql_results: Optional[List[dict]]
    error_message: Optional[str]
    final_response: Optional[str]


class ChinookTextToSQLAgent:
    """Text-to-SQL agent for the Chinook database"""
    
    def __init__(self):
        self.db_connection = None
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.chinook_sql_url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
        
    def initialize_database(self, state: AgentState) -> AgentState:
        """Initialize the in-memory SQLite database with Chinook data"""
        try:
            # Fetch the SQL file
            response = requests.get(self.chinook_sql_url)
            response.raise_for_status()
            sql_content = response.text
            
            # Create in-memory database
            self.db_connection = sqlite3.connect(":memory:")
            cursor = self.db_connection.cursor()
            
            # Execute the SQL to create and populate the database
            cursor.executescript(sql_content)
            self.db_connection.commit()
            
            state["database_initialized"] = True
            return state
            
        except Exception as e:
            state["error_message"] = f"Failed to initialize database: {str(e)}"
            state["database_initialized"] = False
            return state
    
    def extract_schema_info(self, state: AgentState) -> AgentState:
        """Extract comprehensive schema information from the database"""
        if not state.get("database_initialized", False):
            state["error_message"] = "Database not initialized"
            return state
            
        try:
            cursor = self.db_connection.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = cursor.fetchall()
            
            schema_info = "CHINOOK DATABASE SCHEMA:\n\n"
            
            for (table_name,) in tables:
                schema_info += f"Table: {table_name}\n"
                
                # Get table info (columns, types, etc.)
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                schema_info += "Columns:\n"
                for col in columns:
                    col_id, name, data_type, not_null, default_val, pk = col
                    pk_info = " (PRIMARY KEY)" if pk else ""
                    null_info = " NOT NULL" if not_null else ""
                    schema_info += f"  - {name}: {data_type}{null_info}{pk_info}\n"
                
                # Get foreign key info
                cursor.execute(f"PRAGMA foreign_key_list({table_name});")
                fks = cursor.fetchall()
                
                if fks:
                    schema_info += "Foreign Keys:\n"
                    for fk in fks:
                        fk_id, seq, table, from_col, to_col, on_update, on_delete, match = fk
                        schema_info += f"  - {from_col} -> {table}({to_col})\n"
                
                schema_info += "\n"
            
            # Add some sample data info for key tables
            sample_tables = ['Artist', 'Album', 'Track', 'Customer', 'Invoice']
            schema_info += "SAMPLE DATA (first 3 rows):\n\n"
            
            for table in sample_tables:
                if any(t[0] == table for t in tables):
                    cursor.execute(f"SELECT * FROM {table} LIMIT 3;")
                    rows = cursor.fetchall()
                    cursor.execute(f"PRAGMA table_info({table});")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    schema_info += f"{table}:\n"
                    for row in rows:
                        row_data = dict(zip(columns, row))
                        schema_info += f"  {row_data}\n"
                    schema_info += "\n"
            
            state["schema_info"] = schema_info
            return state
            
        except Exception as e:
            state["error_message"] = f"Failed to extract schema: {str(e)}"
            return state
    
    def generate_sql(self, state: AgentState) -> AgentState:
        """Generate SQL query from natural language using LLM"""
        if not state.get("schema_info"):
            state["error_message"] = "Schema information not available"
            return state
        
        # Extract user query from the last human message
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not human_messages:
            state["error_message"] = "No user query found"
            return state
        
        user_query = human_messages[-1].content
        state["user_query"] = user_query
        
        # Check if query is relevant to database operations
        irrelevant_patterns = [
            r'\b(weather|news|sports|politics|cooking|recipe)\b',
            r'\b(how are you|hello|hi|goodbye|bye)\b',
            r'\b(what is|who is|when is|where is)\b(?!.*\b(album|artist|track|customer|invoice|genre|playlist|employee)\b)',
        ]
        
        query_lower = user_query.lower()
        for pattern in irrelevant_patterns:
            if re.search(pattern, query_lower) and not any(word in query_lower for word in ['album', 'artist', 'track', 'customer', 'invoice', 'genre', 'playlist', 'employee', 'music', 'song', 'band']):
                state["final_response"] = "I don't know the answer to that. I can only help with questions about the music database."
                return state
        
        # Create prompt for SQL generation
        sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert. Given a database schema and a natural language question, generate a precise SQL query.

IMPORTANT RULES:
1. Only generate SELECT queries - no INSERT, UPDATE, DELETE, or DDL statements
2. Use proper SQL syntax for SQLite
3. Join tables appropriately based on foreign key relationships
4. Use LIMIT when appropriate to avoid overwhelming results
5. If the question cannot be answered with the available tables, respond with "IRRELEVANT_QUERY"
6. Always use proper table and column names as shown in the schema
7. Use aggregate functions (COUNT, SUM, AVG, etc.) when appropriate
8. Handle case-insensitive searches with LOWER() function when needed

Database Schema:
{schema}

Generate a SQL query for this question. Return ONLY the SQL query, no explanations."""),
            ("human", "{question}")
        ])
        
        try:
            response = self.llm.invoke(sql_prompt.format(
                schema=state["schema_info"],
                question=user_query
            ))
            
            generated_sql = response.content.strip()
            
            # Check if query was deemed irrelevant
            if "IRRELEVANT_QUERY" in generated_sql:
                state["final_response"] = "I don't know the answer to that. I can only help with questions about the music database."
                return state
            
            # Clean up the SQL (remove markdown formatting if present)
            generated_sql = re.sub(r'```sql\n?', '', generated_sql)
            generated_sql = re.sub(r'```\n?', '', generated_sql)
            generated_sql = generated_sql.strip()
            
            state["generated_sql"] = generated_sql
            return state
            
        except Exception as e:
            state["error_message"] = f"Failed to generate SQL: {str(e)}"
            return state
    
    def execute_sql(self, state: AgentState) -> AgentState:
        """Execute the generated SQL query against the database"""
        if not state.get("generated_sql"):
            state["error_message"] = "No SQL query to execute"
            return state
        
        try:
            cursor = self.db_connection.cursor()
            
            # Security check - only allow SELECT statements
            sql_upper = state["generated_sql"].upper().strip()
            if not sql_upper.startswith("SELECT"):
                state["error_message"] = "Only SELECT queries are allowed"
                return state
            
            # Execute the query
            cursor.execute(state["generated_sql"])
            results = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            # Convert results to list of dictionaries
            result_dicts = []
            for row in results:
                result_dicts.append(dict(zip(column_names, row)))
            
            state["sql_results"] = result_dicts
            return state
            
        except Exception as e:
            state["error_message"] = f"SQL execution error: {str(e)}"
            return state
    
    def generate_response(self, state: AgentState) -> AgentState:
        """Generate natural language response from SQL results"""
        if state.get("final_response"):
            # Already have a final response (e.g., for irrelevant queries)
            return state
        
        if state.get("error_message"):
            state["final_response"] = "I don't know the answer to that. I can only help with questions about the music database."
            return state
        
        if not state.get("sql_results"):
            state["final_response"] = "I don't know the answer to that. I can only help with questions about the music database."
            return state
        
        try:
            user_query = state.get("user_query", "")
            sql_results = state["sql_results"]
            generated_sql = state.get("generated_sql", "")
            
            # If no results found
            if not sql_results:
                state["final_response"] = "I found no results for your query in the database."
                return state
            
            # Create prompt for response generation
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that converts SQL query results into natural language responses.

RULES:
1. Provide a clear, concise answer based on the SQL results
2. Use natural language that directly answers the user's question
3. If there are many results, summarize appropriately or show the most relevant ones
4. Include specific numbers, names, and details from the results
5. Don't mention SQL or technical details in your response
6. Keep the response conversational and helpful

User's original question: {question}
SQL query executed: {sql_query}
Results: {results}

Provide a natural language response that answers the user's question based on these results."""),
                ("human", "Please provide a natural language response for the above query and results.")
            ])
            
            # Limit results for response generation if too many
            display_results = sql_results[:20] if len(sql_results) > 20 else sql_results
            
            response = self.llm.invoke(response_prompt.format(
                question=user_query,
                sql_query=generated_sql,
                results=json.dumps(display_results, indent=2)
            ))
            
            final_response = response.content.strip()
            
            # Add count info if results were truncated
            if len(sql_results) > 20:
                final_response += f"\n\n(Showing first 20 of {len(sql_results)} total results)"
            
            state["final_response"] = final_response
            return state
            
        except Exception as e:
            state["final_response"] = "I don't know the answer to that. I can only help with questions about the music database."
            return state
    
    def should_continue(self, state: AgentState) -> str:
        """Determine the next step in the workflow"""
        if not state.get("database_initialized", False):
            return "initialize_db"
        elif not state.get("schema_info"):
            return "extract_schema"
        elif not state.get("user_query"):
            return "generate_sql"
        elif state.get("final_response") and "I don't know" in state["final_response"]:
            return "finalize"
        elif not state.get("generated_sql"):
            return "generate_sql"
        elif not state.get("sql_results") and not state.get("error_message"):
            return "execute_sql"
        elif not state.get("final_response"):
            return "generate_response"
        else:
            return "finalize"
    
    def finalize_response(self, state: AgentState) -> AgentState:
        """Add the final response to messages and return"""
        final_response = state.get("final_response", "I don't know the answer to that.")
        
        # Add AI response to messages
        state["messages"].append(AIMessage(content=final_response))
        
        return state


def create_agent():
    """Create and return the compiled LangGraph agent"""
    agent = ChinookTextToSQLAgent()
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("initialize_db", agent.initialize_database)
    workflow.add_node("extract_schema", agent.extract_schema_info)
    workflow.add_node("generate_sql", agent.generate_sql)
    workflow.add_node("execute_sql", agent.execute_sql)
    workflow.add_node("generate_response", agent.generate_response)
    workflow.add_node("finalize", agent.finalize_response)
    
    # Set entry point
    workflow.set_entry_point("initialize_db")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "initialize_db",
        agent.should_continue,
        {
            "initialize_db": "initialize_db",
            "extract_schema": "extract_schema",
            "generate_sql": "generate_sql",
            "execute_sql": "execute_sql",
            "generate_response": "generate_response",
            "finalize": "finalize"
        }
    )
    
    workflow.add_conditional_edges(
        "extract_schema",
        agent.should_continue,
        {
            "initialize_db": "initialize_db",
            "extract_schema": "extract_schema",
            "generate_sql": "generate_sql",
            "execute_sql": "execute_sql",
            "generate_response": "generate_response",
            "finalize": "finalize"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_sql",
        agent.should_continue,
        {
            "initialize_db": "initialize_db",
            "extract_schema": "extract_schema",
            "generate_sql": "generate_sql",
            "execute_sql": "execute_sql",
            "generate_response": "generate_response",
            "finalize": "finalize"
        }
    )
    
    workflow.add_conditional_edges(
        "execute_sql",
        agent.should_continue,
        {
            "initialize_db": "initialize_db",
            "extract_schema": "extract_schema",
            "generate_sql": "generate_sql",
            "execute_sql": "execute_sql",
            "generate_response": "generate_response",
            "finalize": "finalize"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_response",
        agent.should_continue,
        {
            "initialize_db": "initialize_db",
            "extract_schema": "extract_schema",
            "generate_sql": "generate_sql",
            "execute_sql": "execute_sql",
            "generate_response": "generate_response",
            "finalize": "finalize"
        }
    )
    
    # Add edge from finalize to END
    workflow.add_edge("finalize", END)
    
    # Compile the graph
    return workflow.compile()


# Export the compiled graph as 'app'
app = create_agent()
