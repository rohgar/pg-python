from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from PIL import Image
import io
from my_basic_tool_node import MyBasicToolNode
from my_chain import MyChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from my_tools import MyTools


# represents user inputs/preferences
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


class MyGraph:

    def __init__(self, my_tools: MyTools):
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
            # e.g., "tools": "my_tools"
            {"ftools": "f_tools", END: END},
        )

        # add tool node:
        tools = my_tools.tools
        # graph_builder.add_edge("f_tools", "call_model")
        graph_builder.add_node("f_tools", MyBasicToolNode(tools))

        # add output:
        # graph_builder.add_edge("call_model", END)

        self.graph = graph_builder.compile(checkpointer=MemorySaver())  # Add memory
        my_chain = MyChain(my_tools=my_tools)
        self.model = my_chain.model
        self.chain = my_chain.chain

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
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "ftools"
        return END

    def call_model(self, state: State):
        # trimmed_messages = trimmer.invoke(state["messages"])
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer all questions to the best of your ability in {language}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt_template.invoke(state)
        # response = self.chain.invoke(prompt)
        response = self.model.invoke(prompt)
        return {"messages": response}