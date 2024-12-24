from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains.sequential import SequentialChain
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
# from langchain.chains.sequential import SimpleSequentialChain
# instead of 'from langchain.chains import LLMChain'
# from langchain_core.runnables.base import RunnableSequence

# simple chains are good for a single input, single output

llm = ChatOllama(model="llama3.2", temperature=0.9)

prompt_01 = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?. Provide just the name and no other text."
)
chain_01 = prompt_01 | llm | StrOutputParser() | (lambda x: {"company_name": x})

prompt_02 = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following company: {company_name}. Just provide the description, and no other text."
)
chain_02 = prompt_02 | llm

product = "Queen Size Sheet Set"

# response = chain_01.invoke(product)
# print(f"response = {response.content}")
# print("---")

runnable_seq = chain_01 | chain_02

response = runnable_seq.invoke({"product": product})
print(response.content)
