from langchain_core.messages import BaseMessage
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from my_basic_tool_node import MyBasicToolNode
from PIL import Image
from typing import Sequence
from typing_extensions import Annotated, TypedDict
import io
import sys
sys.path.append("..")
from my_tools import wikipedia_tool, duckduckgo_tool

MODEL = ChatOllama(model="llama3.2", temperature=0.2)


# represents user inputs/preferences
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class MyGraph:

    def __init__(self):
        graph_builder = StateGraph(state_schema=State)

        # where to start its work each time we run our graph. This is
        # edge because we start with an input to the node we will define
        graph_builder.add_edge(START, "call_model")
        # add node:
        graph_builder.add_node("call_model", self.call_model)

        # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
        # it is fine directly responding. This conditional routing defines the main agent loop.
        graph_builder.add_conditional_edges(
            "call_model",
            self.route_tools,
            # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
            # It defaults to the identity function, but if you
            # want to use a node named something else apart from "tools",
            # You can update the value of the dictionary to something else
            {"ftools": "f_tools", END: END},
        )

        # tools
        tools = [ wikipedia_tool, duckduckgo_tool ]

        # add tool node:
        graph_builder.add_edge("f_tools", "call_model")
        graph_builder.add_node("f_tools", MyBasicToolNode(tools))

        # add output:
        graph_builder.add_edge("call_model", END)

        # self.model_w_tools = MODEL
        self.model_w_tools = MODEL.bind_tools(tools)
        self.graph = graph_builder.compile(checkpointer=MemorySaver())  # Add memory


    def display_graph(self):
        try:
            img_data = self.graph.get_graph().draw_mermaid_png()
            Image.open(io.BytesIO(img_data)).show()
        except Exception as e:
            print('Could not display graph: ', repr(e))
            # This requires some extra dependencies and is optional

    def route_tools(self, state: State):
        """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")

        print(f"ai_message = {ai_message}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "ftools"
        return END

    def call_model(self, state: State, config: dict):
        # print(f"state = {state}")
        response = [self.model_w_tools.invoke(state["messages"])]
        return {"messages": response}

