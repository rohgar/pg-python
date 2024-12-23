# from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


from langchain_core.messages import HumanMessage

model = ChatOllama(model="llama3.2", temperature=0.2)

print(model.invoke([HumanMessage(content="What is the capital of France?")]))
print(model.invoke([HumanMessage(content="What are some tourist attractions there?")]))

print('-----------------------------------------------------------------------------')

# conversation
print(model.invoke(
    [
        HumanMessage(content="What is the capital of France?"),
        AIMessage(content="The capital of France is Paris!"),
        HumanMessage(content="What are some tourist attractions there?"),
    ]
))

print('-----------------------------------------------------------------------------')

# conversation using langgraph:

workflow = StateGraph(state_schema=MessagesState)         # Define a new graph

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

workflow.add_edge(START, "model")                         # Define the (single) node in the graph
workflow.add_node("model", call_model)

app = workflow.compile(checkpointer=MemorySaver())               # Add memory

config = {"configurable": {"thread_id": "abc123"}}


input_messages = [HumanMessage("What is the capital of France?")]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

input_messages = [HumanMessage("What are some tourist attractions there?")]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
