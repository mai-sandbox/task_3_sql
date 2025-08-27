# Text-to-SQL LangGraph Agent

A LangGraph-based agent that converts natural language queries to SQL, executes them against the Chinook database, and returns natural language responses.

## Features

- **Natural Language to SQL**: Converts user questions into SQL queries
- **Chinook Database**: Uses the Chinook sample database (music store data)
- **In-Memory Database**: Automatically fetches and sets up the database in memory
- **Natural Language Responses**: Returns query results in readable format
- **Error Handling**: Gracefully handles invalid queries or questions outside the database scope

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Direct invocation:
```python
from agent import app
from langchain_core.messages import HumanMessage

initial_state = {
    "messages": [HumanMessage("How many customers are there?")]
}
result = app.invoke(initial_state)
print(result["messages"][-1].content)
```

### Testing:
```bash
python test_agent.py
```

## Example Queries

- "How many customers are there?"
- "What are the top 5 best selling albums?"
- "Which artist has the most albums?"
- "List all genres in the database"
- "What's the total revenue from all invoices?"
- "Which employee has been with the company the longest?"
- "Show me customers from Canada"

## Architecture

The agent uses a LangGraph state graph with three main nodes:

1. **generate_sql**: Converts natural language to SQL using LLM
2. **execute_sql**: Executes the SQL query against the in-memory database
3. **generate_response**: Generates a natural language response from the results

The Chinook database schema includes tables for:
- Artists, Albums, Tracks
- Customers, Employees
- Invoices, InvoiceLines
- Genres, MediaTypes, Playlists

## Files

- `agent.py` - Main agent implementation
- `langgraph.json` - LangGraph configuration
- `test_agent.py` - Test script with example queries
- `requirements.txt` - Python dependencies
