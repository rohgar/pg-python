from langchain.chains import ConversationChain
from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate


model = OllamaLLM(model="llama3.2", temperature=0.0)

response_str = model.invoke("What is the capital of France?")
print(f"response_str = {response_str}")

response_str = model.invoke("What is its population?")
print(f"response_str = {response_str}\n")