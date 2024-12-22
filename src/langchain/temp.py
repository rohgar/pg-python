# from langchain_core.messages import SystemMessage, trim_messages
# from langchain_ollama.llms import OllamaLLM
# from langchain_core.messages import HumanMessage
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# model = OllamaLLM(model="llama3.2", temperature=0.0)


# trimmer = trim_messages(
#     max_tokens=65,
#     strategy="last",
#     token_counter=model,
#     include_system=True,
#     allow_partial=False,
#     start_on="human",
# )

# messages = [
#     SystemMessage(content="you're a good assistant"),
#     HumanMessage(content="hi! I'm bob"),
#     AIMessage(content="hi!"),
#     HumanMessage(content="I like vanilla ice cream"),
#     AIMessage(content="nice"),
#     HumanMessage(content="whats 2 + 2"),
#     AIMessage(content="4"),
#     HumanMessage(content="thanks"),
#     AIMessage(content="no problem!"),
#     HumanMessage(content="having fun?"),
#     AIMessage(content="yes!"),
# ]

# trimmer.invoke(messages)

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "hi"}, {"output": "whats up"})

memory.load_memory_variables({})