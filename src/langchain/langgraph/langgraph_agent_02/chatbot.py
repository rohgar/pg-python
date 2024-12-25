from langchain_core.messages import HumanMessage
import os
from my_graph import MyGraph
from langchain_core.messages import AIMessage, ToolMessage


# Ref: https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-5-manually-updating-the-state


# def process_input(user_text: str, user_language: str):
#     input_messages = [HumanMessage(user_text)]
#     state = {
#         "messages": input_messages,
#         "language": user_language
#     }
#     output = my_graph.graph.invoke(state, config)
#     return output["messages"][-1]


mg = MyGraph()
# mg.display_graph()

# exit(0)

print("\nType 'exit | quit' to quit the application.\n")
user_language = 'English' #input("Preferred conversation language: ")
# print("\n")

# config = {"configurable": {"thread_id": "abc123"}}

# while True:
#     # Prompt the user for input
#     user_input = "Whats the weather in SF right now?"
#     user_input = input("User: ")

#     # Check if user wants to exit
#     user_input_lower = user_input.lower().strip()
#     if user_input_lower == "exit" or user_input_lower == "quit":
#         print("Exiting...")
#         break


#     # The config is the **second positional argument** to stream() or invoke()!
#     output = mg.graph.invoke(
#         {"messages": [("user", user_input)]}, config, stream_mode="values"
#     )
#     response = output["messages"][-1]
#     for item in output:
#         print(f"System: {response.content}\n---")


user_input = """I need some expert guidance for building an AI agent.
Could you request assistance for me?
"""
config = {"configurable": {"thread_id": "1"}}
events = mg.graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

snapshot = mg.graph.get_state(config)
existing_message = snapshot.values["messages"][-1]
existing_message.pretty_print()

# exit(0)

# Interruption happens here
# if we pass in None, then the chatbot goes ahead and responds
# > It seems like I couldn't get assistance directly from the tool. Let me try again with a different approach.

human_input = input("Human: ")
if not human_input:
    message_dict = None
else:
    new_messages = [
        # The LLM API expects some ToolMessage to match its tool call. We'll satisfy that here.
        ToolMessage(content=human_input, tool_call_id=existing_message.tool_calls[0]["id"]),
        # And then directly "put words in the LLM's mouth" by populating its response.
        AIMessage(content=human_input),
    ]
    message_dict = {"messages": new_messages}
    mg.graph.update_state(
        # Which state to update
        config,
        # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
        # to the existing state. We will review how to update existing messages in the next section!
        {"messages": new_messages},
    )

print("\n\nLast 2 messages;")
print(mg.graph.get_state(config).values["messages"][-2:])
print("---")

# However, if we pass a new_message, then that is used by the chatbot:
# Eg. enter "LangGraph is a library for building stateful, multi-actor applications with LLMs."
events = mg.graph.stream(message_dict, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


# Do the same above programmatically
# answer = (
#     "LangGraph is a library for building stateful, multi-actor applications with LLMs."
# )
# new_messages = [
#     # The LLM API expects some ToolMessage to match its tool call. We'll satisfy that here.
#     ToolMessage(content=answer, tool_call_id=existing_message.tool_calls[0]["id"]),
#     # And then directly "put words in the LLM's mouth" by populating its response.
#     AIMessage(content=answer),
# ]

exit(0)
# new_messages[-1].pretty_print()
# mg.graph.update_state(
#     # Which state to update
#     config,
#     # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
#     # to the existing state. We will review how to update existing messages in the next section!
#     {"messages": new_messages},
# )

print("\n\nLast 2 messages;")
print(mg.graph.get_state(config).values["messages"][-2:])

