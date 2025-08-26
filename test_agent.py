#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent import app

load_dotenv()

def test_agent():
    """Test the SQL agent with various queries"""
    
    test_queries = [
        "How many customers are there?",
        "What are the top 5 albums by number of tracks?",
        "Show me all employees who report to Andrew Adams",
        "What is the total revenue from all invoices?",
        "List the top 3 customers by total purchase amount",
        "What is the weather today?",  # This should return "I don't know"
    ]
    
    print("=" * 60)
    print("Testing LangGraph SQL Agent")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        try:
            result = app.invoke(initial_state)
            
            if result.get("messages"):
                ai_response = result["messages"][-1].content
                print(f"💬 Response: {ai_response}")
            
            if result.get("sql_query") and result["sql_query"] != "UNABLE_TO_ANSWER":
                print(f"\n🔍 SQL Generated: {result['sql_query']}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_agent()