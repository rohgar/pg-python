import warnings
# duckduckgo-search latest package warning: UserWarning: 'api' backend is deprecated, using backend='auto'
warnings.filterwarnings('ignore', message="'api' backend is deprecated")

from langchain_core.messages import HumanMessage
from agent import Agent
from langchain_ollama import ChatOllama

# we will use sqllite to save state
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List

import sys
sys.path.append("..")
from my_tools import get_tools

# Reference: https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/7/essay-writer

def print_header(title: str):
    print('===============================================')
    print(title)
    print('===============================================')

tools = get_tools(include_tavily=True)
model = ChatOllama(model="llama3.2", temperature=0.2)


# session - this will allow us to have multiple conversations with different
# user simultaneously
thread = {"configurable": {"thread_id": "2"}}


if __name__ == "__main__":
    # we use an in-memory SQLite database for our checkpoints
    with SqliteSaver.from_conn_string(":memory:") as memory:
        agent = Agent(model, checkpointer=memory)
        # agent.display_graph()

        thread = {"configurable": {"thread_id": "1"}}
        for s in agent.graph.stream({
            'task': "what is the difference between langchain and langsmith",
            "max_revisions": 2,
            "revision_number": 1,
        }, thread):
            print(s)
            print('---')








