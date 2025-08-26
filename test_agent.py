from agent import app
from langchain_core.messages import HumanMessage

def test_agent():
    """Test the text-to-SQL agent with various queries."""
    
    test_queries = [
        "How many customers are in the database?",
        "What are the top 5 albums by number of tracks?",
        "Show me all employees who report to Andrew Adams",
        "What is the total sales amount for the year 2009?",
        "What's the weather like today?",  # This should return "I don't know"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        initial_state = {
            "messages": [HumanMessage(query)]
        }
        
        try:
            result = app.invoke(initial_state)
            final_message = result["messages"][-1].content
            print(f"Response: {final_message}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_agent()