from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


model = OllamaLLM(model="llama3.2", temperature=0.0)


prompt_template = ChatPromptTemplate([
    ("system", "Can you answer this question: "),
    ("user", "{topic}")
])

prompt = prompt_template.invoke({"topic": "cats"})

chain = prompt_template | model

# response_str = chain.invoke(prompt)
# print(f"response_str = {response_str}")


# ------------------------------
# Using user input, we can use the `Human Message Object`
# ------------------------------

# the key should match the input variable defined chat prompt template
prompt = prompt_template.invoke({"topic": [HumanMessage(content="What is the capital of France?")]})

response_str = chain.invoke(prompt)
print(f"response_str = {response_str}\n")

prompt = prompt_template.invoke({"topic": [HumanMessage(content="What is its population?")]})

response_str = chain.invoke(prompt)
print(f"response_str explaination = {response_str}")
print("------\n")


# ------------------------------
# Memory
# ------------------------------

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage
from langchain_core.messages import SystemMessage, trim_messages

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable": {"session_id": "abc2a1"}}

response = with_message_history.stream(
    prompt_template.format(topic="What is the capital of France?"),
    config=config,
)

print(f"1) response {type(response)} = {response}")


response = with_message_history.stream(
    prompt_template.format(topic="What is the language spoken there?"),
    config=config,
)
print(f"2) response {type(response)} = {response}")


response = with_message_history.stream(
    prompt_template.format(topic="Can you name some popular attractions there?"),
    config=config,
)
print(f"3) response_str = {response}")

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)