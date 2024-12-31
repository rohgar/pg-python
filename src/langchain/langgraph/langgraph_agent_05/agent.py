from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    HumanMessage,
    trim_messages,
)
from typing import TypedDict, Annotated, List
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
from tavily import TavilyClient
import os
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed:

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: List[str]


class Agent:

    def __init__(self, model, checkpointer=None):
        self.model = model
        graph_builder = StateGraph(AgentState)
        # add nodes

        graph_builder.add_node("planner", self.plan_node)
        graph_builder.add_node("generate", self.generation_node)
        graph_builder.add_node("reflect", self.reflection_node)
        graph_builder.add_node("research_plan", self.research_plan_node)
        graph_builder.add_node("research_critique", self.research_critique_node)

        graph_builder.add_conditional_edges(
            "generate",
            self.should_continue,
            {END: END, "reflect": "reflect"}
        )
        graph_builder.add_edge("planner", "research_plan")
        graph_builder.add_edge("research_plan", "generate")
        graph_builder.add_edge("reflect", "research_critique")
        graph_builder.add_edge("research_critique", "generate")


        graph_builder.set_entry_point("planner")
        self.graph = graph_builder.compile(
            checkpointer=checkpointer,
        )


    def display_graph(self):
        try:
            img_data = self.graph.get_graph().draw_mermaid_png()
            Image.open(io.BytesIO(img_data)).show()
        except Exception as e:
            print('Could not display graph: ', repr(e))
            # This requires some extra dependencies and is optional

    def plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ]
        response = self.model.invoke(messages)
        return {"plan": response.content}

    def research_plan_node(self, state: AgentState):
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ])
        content = []
        if 'content' in state:
            content = state['content']
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}

    def generation_node(self, state: AgentState):
        content = "\n\n".join(state['content'] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
        messages = [
            SystemMessage(
                content=WRITER_PROMPT.format(content=content)
            ),
            user_message
            ]
        response = self.model.invoke(messages)
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 1) + 1
        }

    def reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content=REFLECTION_PROMPT),
            HumanMessage(content=state['draft'])
        ]
        response = self.model.invoke(messages)
        return {"critique": response.content}

    def research_critique_node(self, state: AgentState):
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state['critique'])
        ])
        content = state['content'] or []
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}

    def should_continue(self, state):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"
