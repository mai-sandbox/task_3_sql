# Text-to-SQL Agent for Chinook Database

## Setup

1. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

2. Add your Anthropic API key to the `.env` file:
   ```
   ANTHROPIC_API_KEY=your_actual_api_key_here
   ```

## Usage

### Using the Agent Programmatically

```python
from agent import app
from langchain_core.messages import HumanMessage

# Create initial state with your query
initial_state = {
    "messages": [HumanMessage("How many customers are in the database?")]
}

# Invoke the agent
result = app.invoke(initial_state)

# Get the response
response = result["messages"][-1].content
print(response)
```

### Example Queries

The agent can answer questions about:
- Customer data: "How many customers are from Canada?"
- Sales data: "What was the total revenue in 2009?"
- Music catalog: "Which artist has the most albums?"
- Employee data: "List all employees and their titles"
- Track information: "What are the longest songs in the database?"

### Invalid Queries

The agent will respond with "I don't know the answer" for:
- Questions unrelated to the database (e.g., "What's the weather?")
- Questions that cannot be answered with the available data
- Requests outside the scope of the Chinook music database

## Testing

Run the test script to verify the agent works:
```bash
python test_agent.py
```

Run the database test to verify Chinook loads correctly:
```bash
python test_db.py
```