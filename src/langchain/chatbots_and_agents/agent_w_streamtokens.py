# from langchain_ollama.llms import OllamaLLM
import asyncio
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

async def process_input(user_text: str):
    input_messages = [HumanMessage(user_text)]
    output = agent_executor.astream_events({"messages": input_messages}, config, version="v1")
    return output


model = ChatOllama(model="llama3.2", temperature=0.2)
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "abc123"}}

async def main():
    while True:
        # Prompt the user for input
        user_text = input("User: ")

        # Check if user wants to exit
        if user_text.lower().strip() == "exit" or user_text.lower().strip() == "quit":
            print("Exiting...")
            break

        # Process the input
        response = await process_input(user_text)

        async for event in response:
            kind = event["event"]
            if kind == "on_chain_start":
                if ( event["name"] == "Agent"):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                    print(
                        f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                    )
            elif kind == "on_chain_end":
                if ( event["name"] == "Agent" ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                    print()
                    print("--")
                    print(
                        f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                    )
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # Empty content in the context of OpenAI means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content
                    print(content, end="|")
            elif kind == "on_tool_start":
                print("--")
                print(
                    f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
                )
            elif kind == "on_tool_end":
                print(f"Done tool: {event['name']}")
                print(f"Tool output was: {event['data'].get('output')}")
                print("--")

asyncio.run(main())
