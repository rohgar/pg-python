import warnings
# duckduckgo-search latest package warning: UserWarning: 'api' backend is deprecated, using backend='auto'
warnings.filterwarnings('ignore', message="'api' backend is deprecated")

from langchain_core.messages import HumanMessage
from agent import Agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver

import sys
sys.path.append("..")
from my_tools import get_tools


# Reference: https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/3/langgraph-components

prompt = """You are a smart research assistant. Use the available tools to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
Give concise answers.
"""

tools = get_tools()
model = ChatOllama(model="llama3.2", temperature=0.2)

# session
thread = {"configurable": {"thread_id": "2"}}


def ask_query(query: str):
    messages = [HumanMessage(content=query)]

    # for event in agent.graph.stream({"messages": messages}, thread):
    #     for v in event.values():
    #         print(v)

    result = agent.graph.invoke({"messages": messages}, thread)
    print_result(result)


# result is type: langgraph.pregel.io.AddableValuesDict
def print_result(result):
    print(f"result = {result['messages'][-1].content}\n---\n")


with SqliteSaver.from_conn_string(":memory:") as memory:
    agent = Agent(model, tools, system_msg_prompt=prompt, checkpointer=memory)
    # agent.display_graph()

    query = "Whats the weather in SF?"
    ask_query(query)

    query = "What is the weather in SF and LA?"
    ask_query(query)

    query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
    What is the GDP of that state? Answer each question."
    ask_query(query)
