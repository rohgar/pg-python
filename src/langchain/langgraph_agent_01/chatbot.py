from langchain_core.messages import HumanMessage
import os
from my_graph import MyGraph
from my_tools import MyTools


mt = MyTools()
mg = MyGraph(my_tools=mt)
# mg.display_graph()

print("\nType 'exit | quit' to quit the application.\n")
user_language = 'English' #input("Preferred conversation language: ")
# print("\n")

config = {"configurable": {"thread_id": "abc123"}}

while True:
    # Prompt the user for input
    user_input = "Whats the weather in SF right now?"
    user_input = input("User: ")

    # Check if user wants to exit
    user_input_lower = user_input.lower().strip()
    if user_input_lower == "exit" or user_input_lower == "quit":
        print("Exiting...")
        break


    # The config is the **second positional argument** to stream() or invoke()!
    output = mg.graph.invoke(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    response = output["messages"][-1]
    for item in output:
        print(f"System: {response.content}\n---")
