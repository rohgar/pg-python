from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
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
import operator
from uuid import uuid4


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:

    def __init__(self, model, tools, system_msg_prompt="", checkpointer=None):
        # this is the prompt
        self.system_msg_prompt = system_msg_prompt

        graph_builder = StateGraph(AgentState)
        # add nodes
        graph_builder.add_node("llm", self.call_model)
        graph_builder.add_node("action", self.take_action)
        # add conditional edge from the "llm" node to the function where we want to check
        # if we want to take the action. The third dict argument is the mapping based on the return
        # of the function
        graph_builder.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        # add a regular edge
        graph_builder.add_edge("action", "llm")
        graph_builder.set_entry_point("llm")
        self.graph = graph_builder.compile(
            checkpointer=checkpointer,
        )

        # create a dict of the toolname to the tool
        self.tools = {t.name: t for t in tools}
        # bind the tools to the model so that the model can call the tools
        self.model = model.bind_tools(tools)

    # Function to call the llm. Takes the messages from the state, and invokes the
    # model using the messages, and returns a dictionary of that message.
    # This new message will be added to the state, based on our Annotation in the
    # `messages` attribute of the `AgentState` class.
    def call_model(self, state: AgentState):
        messages = state['messages']
        if self.system_msg_prompt:
            messages = [SystemMessage(content=self.system_msg_prompt)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    # Function to check whether there is an action present if we want to go to
    # the tool/action node (for the conditional edge).
    # This function checks the llm response i.e. the last message  for the attribute
    # `tool_calls`. If this attribute is set, then we want to take the action.
    def exists_action(self, state: AgentState):
        # print(f'agent.exists_action() state:\n{state}\n')
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    # Function to take the action determined by exists_action. If the control comes
    # to this function, it means that the attribute `tool_calls` will be present, which
    # will contains the list of tools that need to be called, that we iterated over.
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            tool_name = t['name']
            tool = self.tools[tool_name]
            result = tool.invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        # print("Back to the model!")
        return {'messages': results}

    def display_graph(self):
        try:
            img_data = self.graph.get_graph().draw_mermaid_png()
            Image.open(io.BytesIO(img_data)).show()
        except Exception as e:
            print('Could not display graph: ', repr(e))
            # This requires some extra dependencies and is optional

