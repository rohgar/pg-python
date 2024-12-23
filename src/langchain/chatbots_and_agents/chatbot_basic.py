# from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage



def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

def process_input(user_text: str):
    input_messages = [HumanMessage(user_text)]
    output = app.invoke({"messages": input_messages}, config)
    return output["messages"][-1]

model = ChatOllama(model="llama3.2", temperature=0.2)

workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")  # Define the (single) node in the graph
workflow.add_node("model", call_model)

app = workflow.compile(checkpointer=MemorySaver())  # Add memory

config = {"configurable": {"thread_id": "abc123"}}

print("Type 'exit|quit' to quit the application.\n")
while True:
    # Prompt the user for input
    user_text = input("User: ")

    # Check if user wants to exit
    if user_text.lower().strip() == "exit" or user_text.lower().strip() == "quit":
        print("Exiting...")
        break

    # Process the input
    response = process_input(user_text)

    # Display the response
    print(f"System: {response.content}\n---")