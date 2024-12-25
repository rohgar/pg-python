from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain_ollama import ChatOllama
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings.embeddings import Embeddings
from IPython.display import display, Markdown

llm = ChatOllama(model="llama3.2", temperature=0.9)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


loader = CSVLoader(file_path='./resources/OutdoorClothingCatalog_1000.csv')
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embedding_model
).from_loaders([loader])

query = "Please list all your shirts with sun protection in a table in \
markdown and summarize each one."

response = index.query(query, llm=llm)
print(f"response = {response}")

