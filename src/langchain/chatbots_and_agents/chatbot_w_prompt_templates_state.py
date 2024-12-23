# from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from typing import Sequence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, trim_messages
from PIL import Image
import io


# Since now we have user messages and the user specified language as input, we will define a
# State to represent this:

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

def call_model(state: State):
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
    response = chain.invoke(prompt)
    return {"messages": response}

def process_input(user_text: str, user_language: str):
    input_messages = [HumanMessage(user_text)]
    output = graph.invoke({"messages": input_messages, "language": user_language}, config)
    return output["messages"][-1]

model = ChatOllama(model="llama3.2", temperature=0.2)
trimmer = trim_messages(
    max_tokens=500,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)
chain = trimmer | model

graph_builder = StateGraph(state_schema=State)
# where to start its work each time we run our graph. This is
# edge because we start with an input to the node we will define
graph_builder.add_edge(START, "model")
# add the node
graph_builder.add_node("model", call_model)

# compile the graph
graph = graph_builder.compile(checkpointer=MemorySaver())  # Add memory


try:
    img_data = graph.get_graph().draw_mermaid_png()
    Image.open(io.BytesIO(img_data)).show()
except Exception as e:
    print('Could not display graph: ', repr(e))
    # This requires some extra dependencies and is optional
    pass

config = {"configurable": {"thread_id": "abc123"}}

print("Type 'exit|quit' to quit the application.\n")
user_language = input("User (language): ")
while True:
    # Prompt the user for input
    user_text = input("User: ")

    # Check if user wants to exit
    if user_text.lower().strip() == "exit" or user_text.lower().strip() == "quit":
        print("Exiting...")
        break

    # Process the input
    response = process_input(user_text, user_language)

    # Display the response
    print(f"System: {response.content}\n---")