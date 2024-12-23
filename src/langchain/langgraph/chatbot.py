from langchain_core.messages import HumanMessage
import os
from my_graph import MyGraph
from my_tools import MyTools

# create an agent that uses Tavily api to search the internet
# for the user's query.
if not os.getenv('TAVILY_API_KEY'):
    raise Exception("'TAVILY_API_KEY' is not set")


def process_input(user_text: str, user_language: str):
    input_messages = [HumanMessage(user_text)]
    output = graph.invoke(
        {
            "messages": input_messages,
            "language": user_language
        },
        config)
    return output["messages"][-1]


my_tools = MyTools()
my_graph = MyGraph(my_tools=my_tools)
graph = my_graph.graph
my_graph.display_graph()


config = {"configurable": {"thread_id": "abc123"}}


print("Type 'exit|quit' to quit the application.\n")
user_language = input("Preferred conversation language: ")
while True:
    # Prompt the user for input
    user_text = input("User: ")

    # Check if user wants to exit
    user_text_lower = user_text.lower().strip()
    if user_text_lower == "exit" or user_text_lower == "quit":
        print("Exiting...")
        break

    # Process the input
    response = process_input(user_text, user_language)

    # Display the response
    print(f"System: {response.content}\n---")
