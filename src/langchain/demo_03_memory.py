from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


model = OllamaLLM(model="llama3.2", temperature=0.0)


prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

prompt = prompt_template.invoke({"topic": "cats"})

chain = prompt_template | model

# response_str = chain.invoke(prompt)
# print(f"response_str = {response_str}")


# ------------------------------
# Using user input, we can use the `Human Message Object`
# ------------------------------

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

# the key should match the input variable defined chat prompt template
prompt = prompt_template.invoke({"topic": [HumanMessage(content="dogs")]})

response_str = chain.invoke(prompt)
print(f"response_str = {response_str}")


# ------------------------------
# Memory
# ------------------------------

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable": {"session_id": "abc2"}}

response = with_message_history.invoke(
    prompt_template.format(topic="chicken"),
    config=config,
)

print(f"response_str = {response}")