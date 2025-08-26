from agent import app
from langchain_core.messages import HumanMessage

def test_agent():
    """Test the text-to-SQL agent with various queries"""
    
    test_queries = [
        "How many customers are there in the database?",
        "What are the top 5 best selling tracks?",
        "List all albums by the artist 'AC/DC'",
        "What is the total sales amount?",
        "What's the weather today?",  # This should trigger the "don't know" response
        "Show me the employees and their titles"
    ]
    
    print("Testing LangGraph Text-to-SQL Agent\n" + "="*50)
    
    for query in test_queries:
        print(f"\n\nQuery: {query}")
        print("-" * 40)
        
        initial_state = {
            "messages": [HumanMessage(query)],
            "sql_query": "",
            "query_result": "",
            "error": "",
            "schema": ""
        }
        
        try:
            result = app.invoke(initial_state)
            response = result["messages"][-1].content
            print(f"Response: {response}")
            
            if result.get("sql_query") and result["sql_query"] != "INVALID_QUERY":
                print(f"\nGenerated SQL: {result['sql_query']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_agent()