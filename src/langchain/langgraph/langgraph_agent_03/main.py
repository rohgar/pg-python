import warnings
# duckduckgo-search latest package warning: UserWarning: 'api' backend is deprecated, using backend='auto'
warnings.filterwarnings('ignore', message="'api' backend is deprecated")

from langchain_core.messages import HumanMessage
from agent import Agent
from langchain_ollama import ChatOllama

# we will use sqllite to save state
from langgraph.checkpoint.sqlite import SqliteSaver

import sys
sys.path.append("..")
from my_tools import get_tools

# Reference: https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/3/langgraph-components


def ask_query(query: str, stream_result=False):
    # Take the query and create an input for the llm that confirms to the AgentState
    input_dict = None
    if query:
        input_dict = {"messages": [HumanMessage(content=query)]}

    # stream_result will return all the intermediate step results as opposed to the final
    # result. Eg. It will return an AIMessage from the LLM, then ToolMessage from the tool
    # and then an AIMessage which has the final result. Hence the stream is printed in
    # a for loop
    if stream_result:
        for event in agent.graph.stream(input_dict, thread):
            for v in event.values():
                print(v['messages'])
        print("\n")
    else:
        # result is type: langgraph.pregel.io.AddableValuesDict
        result = agent.graph.invoke(input_dict, thread)
        print(f"result = {result['messages'][-1]}\n")
    print("---\n")


prompt = """You are a smart research assistant. Use the available tools to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
Give concise answers.
"""

tools = get_tools()
model = ChatOllama(model="llama3.2", temperature=0.2)

# session - this will allow us to have multiple conversations with different
# user simultaneously
thread = {"configurable": {"thread_id": "2"}}

if __name__ == "__main__":
    # we use an in-memory SQLite database for our checkpoints
    with SqliteSaver.from_conn_string(":memory:") as memory:
        agent = Agent(model, tools, system_msg_prompt=prompt, checkpointer=memory)
        # agent.display_graph()

        query = "Whats the weather in SF?"
        ask_query(query, True)

        query = "How about in LA?"
        ask_query(query)

        query = "Which one of those is warmer?"
        ask_query(query)

        query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
        What is the GDP of that state? Answer each question."
        ask_query(query)
