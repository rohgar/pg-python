from operator import itemgetter
from typing import Literal
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
import asyncio


llm = ChatOllama(model="llama3.2", temperature=0.9)

# Define the prompts we will route to
prompt_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert on animals."),
        ("human", "{input}"),
    ]
)
prompt_2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert on vegetables."),
        ("human", "{input}"),
    ]
)
prompt_3 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert on physics."),
        ("human", "{input}"),
    ]
)

# Construct the chains we will route to. These format the input query
# into the respective prompt, run it through a chat model, and cast
# the result to a string.
chain_1 = prompt_1 | llm | StrOutputParser()
chain_2 = prompt_2 | llm | StrOutputParser()
chain_3 = prompt_3 | llm | StrOutputParser()


# Next: define the chain that selects which branch to route to.
# Here we will take advantage of tool-calling features to force
# the output to select one of two desired branches.
route_system = "Route the user's query to either the animal or vegetable expert."
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", route_system),
        ("human", "{input}"),
    ]
)


# Define schema for output:
class RouteQuery(TypedDict):
    """Route query to destination expert."""
    destination: Literal["animal", "vegetable", "physics"]


route_chain = route_prompt | llm.with_structured_output(RouteQuery)


# For LangGraph, we will define the state of the graph to hold the query,
# destination, and final answer.
class State(TypedDict):
    query: str
    destination: RouteQuery
    answer: str


# We define functions for each node, including routing the query:
async def route_query(state: State, config: RunnableConfig):
    destination = await route_chain.ainvoke(state["query"], config)
    return {"destination": destination}


# And one node for each prompt
async def prompt_1(state: State, config: RunnableConfig):
    return {"answer": await chain_1.ainvoke(state["query"], config)}


async def prompt_2(state: State, config: RunnableConfig):
    return {"answer": await chain_2.ainvoke(state["query"], config)}


async def prompt_3(state: State, config: RunnableConfig):
    return {"answer": await chain_3.ainvoke(state["query"], config)}


# We then define logic that selects the prompt based on the classification
def select_node(state: State) -> Literal["prompt_1", "prompt_2", "prompt_3"]:
    if state["destination"] == "animal":
        return "prompt_1"
    elif state["destination"] == "vegetable":
        return "prompt_2"
    else:
        return "prompt_3"


# Finally, assemble the multi-prompt chain. This is a sequence of two steps:
# 1) Select "animal" or "vegetable" via the route_chain, and collect the answer
# alongside the input query.
# 2) Route the input query to chain_1 or chain_2, based on the
# selection.
graph_builder = StateGraph(State)
graph_builder.add_node("route_query", route_query)
graph_builder.add_node("prompt_1", prompt_1)
graph_builder.add_node("prompt_2", prompt_2)
graph_builder.add_node("prompt_3", prompt_3)

graph_builder.add_edge(START, "route_query")
graph_builder.add_conditional_edges("route_query", select_node)
graph_builder.add_edge("prompt_1", END)
graph_builder.add_edge("prompt_2", END)
graph_builder.add_edge("prompt_3", END)
graph = graph_builder.compile()

display_graph = False
if display_graph:
    try:
        from PIL import Image
        import io
        img_data = graph.get_graph().draw_mermaid_png()
        Image.open(io.BytesIO(img_data)).show()
    except Exception as e:
        print('Could not display graph: ', repr(e))


async def run_chat(graph, config):
    print("\nType 'exit | quit' to quit the application.\n")
    while True:
        # Prompt the user for input
        user_input = input("User: ")

        # Check if user wants to exit
        user_input_lower = user_input.lower().strip()
        if user_input_lower in ["exit", "quit"]:
            print("Exiting...")
            break

        # Invoke the graph asynchronously
        output = await graph.ainvoke(
            {"query": [("user", user_input)]}, config, stream_mode="values"
        )

        # Print the system response
        print(f"System: {output}\n---")


config = {"configurable": {"thread_id": "pqr789"}}
asyncio.run(run_chat(graph, config))
