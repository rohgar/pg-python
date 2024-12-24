from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import ToolNode, tools_condition
from PIL import Image
from pydantic import BaseModel
from typing import Sequence
from typing_extensions import Annotated, TypedDict
import io


class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.
    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    request: str


# represents user inputs/preferences
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ask_human: bool



def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def my_human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_human": False,
    }


MODEL = ChatOllama(model="llama3.2", temperature=0.2)
tools = [TavilySearchResults(max_results=1)]
MODEL_WITH_TOOLS = MODEL.bind_tools(tools + [RequestAssistance])

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

        # add tool node:
        graph_builder.add_edge("f_tools", "call_model")
        graph_builder.add_node("f_tools", ToolNode(tools))
        graph_builder.add_node("human", my_human_node)
        graph_builder.add_edge("human", "call_model")

        graph_builder.add_conditional_edges(
            "call_model",
            self.route_tools,
            # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
            # It defaults to the identity function, but if you
            # want to use a node named something else apart from "tools",
            # You can update the value of the dictionary to something else
            # e.g., "tools": "my_tools"
            {"ftools": "f_tools", END: END},
        )

        graph_builder.add_conditional_edges(
            "call_model",
            self.select_next_node,
            # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
            # It defaults to the identity function, but if you
            # want to use a node named something else apart from "tools",
            # You can update the value of the dictionary to something else
            # e.g., "tools": "my_tools"
            {"human": "human", END: END},
        )

        # add output:
        # graph_builder.add_edge("call_model", END)

        self.graph = graph_builder.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["human"],
        )  # Add memory

    def select_next_node(self, state: State):
        if state["ask_human"]:
            return "human"
        # Otherwise, we can route as before
        return tools_condition(state)

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
        response = MODEL_WITH_TOOLS.invoke(state["messages"])
        ask_human = False
        if (
            response.tool_calls
            and response.tool_calls[0]["name"] == RequestAssistance.__name__
        ):
            ask_human = True


        return {"messages": [response], "ask_human": ask_human}

