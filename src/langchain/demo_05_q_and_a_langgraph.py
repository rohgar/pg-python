from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import CSVLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import ChatPromptTemplate


# See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
""")

llm = ChatOllama(model="llama3.2", temperature=0.9)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

loader = CSVLoader(file_path='./resources/OutdoorClothingCatalog_1000.csv')
document = loader.load()
vectorstore = DocArrayInMemorySearch.from_documents(
    document,
    embedding_model
)
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

response = qa_chain.invoke("Please list all your shirts with sun protection in a table in \
markdown and summarize each one.")
print(f"response = {response}")
