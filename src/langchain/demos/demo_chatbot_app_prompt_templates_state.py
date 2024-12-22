from langchain_ollama.llms import OllamaLLM
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
    output = app.invoke({"messages": input_messages, "language": user_language}, config)
    return output["messages"][-1]

model = OllamaLLM(model="llama3.2", temperature=0.2)
trimmer = trim_messages(
    max_tokens=500,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)
chain = trimmer | model

workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")  # Define the (single) node in the graph
workflow.add_node("model", call_model)

app = workflow.compile(checkpointer=MemorySaver())  # Add memory

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