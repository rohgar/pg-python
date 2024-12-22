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

