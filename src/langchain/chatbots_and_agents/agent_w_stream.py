# from langchain_ollama.llms import OllamaLLM
import os
from langchain_ollama import ChatOllama
# Import relevant functionality
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


# Create an agent that uses Tavily api to search the internet for the user's query.
if not os.getenv('TAVILY_API_KEY'):
    raise Exception("'TAVILY_API_KEY' is not set, export it in the environment before running")

def process_input(user_text: str):
    input_messages = [HumanMessage(user_text)]
    output = agent_executor.stream({"messages": input_messages}, config)
    return output


model = ChatOllama(model="llama3.2", temperature=0.2)
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "abc123"}}


while True:
    # Prompt the user for input
    user_text = input("User: ")

    # Check if user wants to exit
    if user_text.lower().strip() == "exit" or user_text.lower().strip() == "quit":
        print("Exiting...")
        break

    # Process the input
    response = process_input(user_text)

    for chunk in response:
        print(chunk)
        print("----")
