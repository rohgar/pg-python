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

# Reference: https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/6/human-in-the-loop

def print_header(title: str):
    print('===============================================')
    print(title)
    print('===============================================')

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
                print(v)
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

tools = get_tools(include_tavily=True)
model = ChatOllama(model="llama3.2", temperature=0.2)

# session - this will allow us to have multiple conversations with different
# user simultaneously
thread = {"configurable": {"thread_id": "2"}}


# --------------------------------
# Interrupt before action, we pass None and let it proceed
# --------------------------------
def scene_01():
    print_header('scene_01')
    query = "Whats the weather in SF?"
    ask_query(query)
    query = None # pass None to interrupt so that it can proceed without human input
    ask_query(query)

# --------------------------------
# Interrupt before action, put in a loop and check for input programatically
# so as to proceed or abort
# --------------------------------
def scene_02():
    print_header('scene_02')
    query = "Whats the weather in SF?"
    ask_query(query)
    while agent.graph.get_state(thread).next:
        print("\n", agent.graph.get_state(thread),"\n")
        _input = input("Proceed? ")
        if _input != "y":
            print("aborting")
            break
        for event in agent.graph.stream(None, thread):
            for v in event.values():
                print(v)


# --------------------------------
# Make a query. Then take the last state, and change it. Eg. we change SF to Louisiana
# Then update the graph with that state and run the query
# --------------------------------
def scene_03():
    print_header('scene_03')
    query = "Whats the weather in SF?"
    ask_query(query)
    last_state = agent.graph.get_state(thread)
    print(f"last_state = {last_state}")
    print('-')
    _id = last_state.values['messages'][-1].tool_calls[0]['id']
    last_state.values['messages'][-1].tool_calls = [
        {'name': 'tavily_search_results_json',
      'args': {'query': 'Louisiana weather'},
      'id': _id}
    ]
    print(f"updated last_state = {last_state}")
    print('-')
    agent.graph.update_state(thread, last_state.values)
    for event in agent.graph.stream(None, thread):
        for v in event.values():
            print(v)

# --------------------------------
# Time travel. We can call get_state_history on the graph to see all previous states
# --------------------------------
def scene_04():
    # run some queries to have a few states
    scene_03()
    # get all the states into an array
    print_header('scene_04')
    states_arr = []
    for state in agent.graph.get_state_history(thread):
        print(state)
        print('--')
        states_arr.append(state)

    # replay the desired state
    to_replay = states_arr[-3]
    for event in agent.graph.stream(None, to_replay.config):
        for k, v in event.items():
            print(v)

if __name__ == "__main__":
    # we use an in-memory SQLite database for our checkpoints
    with SqliteSaver.from_conn_string(":memory:") as memory:
        agent = Agent(model, tools, system_msg_prompt=prompt, checkpointer=memory)
        # agent.display_graph()

        # scene_01()
        # scene_02()
        # scene_03()
        scene_04()






