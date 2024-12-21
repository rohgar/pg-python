from langchain.chains import ConversationChain
from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate


model = OllamaLLM(model="llama3.2", temperature=0.0)

# response_str = model.invoke("What is the capital of France?")
# print(f"response_str = {response_str}")

# response_str = model.invoke("What is its population?")
# print(f"response_str = {response_str}\n")

# ---------------------------------------
# Previous
# ---------------------------------------

from langchain_core.prompts import ChatPromptTemplate


prompt_template_str = """Translate the text that is delimited by triple \
backticks into a style that is {customer_style}.
text: ```{customer_email}```
"""
prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
chain = prompt_template | model

style = """American English in a calm and respectful tone"""

email_content = """Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls with smoothie! \
And to make matters worse, the warranty don't cover the \
cost of cleaning up me kitchen. I need yer help right \
now, matey!
"""

# response_str = chain.invoke(
# 	{"customer_email": email_content, "customer_style": style}
# )
# print(f"response_str ({type(response_str)}) = {response_str}\n")

# ---------------------------------------
# Memory
# ---------------------------------------

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage
# from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOllama



store = {}

# stores an instance of `InMemoryChatMessageHistory` under the resp. session_id
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable": {"session_id": "abc2"}}

message = HumanMessage(
    content=prompt_template.format(
        customer_email=email_content,
        customer_style=style
    )
)

response = with_message_history.invoke(
    "Hello",
    config=config,
)

print(f"response = {response}")