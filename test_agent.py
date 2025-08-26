from agent import app
from langchain_core.messages import HumanMessage


def test_agent():
    """Test the text-to-SQL agent with various queries"""
    
    test_queries = [
        "How many artists are in the database?",
        "What are the top 5 customers by total purchase amount?",
        "List all albums by Led Zeppelin",
        "How many tracks are longer than 5 minutes?",
        "What's the weather today?",  # This should return "don't know"
        "Tell me about the employees in the company",
        "What genres of music are available?",
        "How many invoices were created in 2009?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('-'*60)
        
        try:
            initial_state = {
                "messages": [HumanMessage(query)]
            }
            
            result = app.invoke(initial_state)
            
            if "final_answer" in result:
                print(f"Answer: {result['final_answer']}")
            else:
                last_message = result["messages"][-1]
                print(f"Answer: {last_message.content}")
                
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("Testing LangGraph Text-to-SQL Agent")
    print("="*60)
    test_agent()