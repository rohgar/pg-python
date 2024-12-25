from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain_ollama import ChatOllama
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_huggingface import HuggingFaceEmbeddings
from IPython.display import display, Markdown

# This script does the same as demo_05_q_and_a.py, but shows different ways of doing the same thing.


llm = ChatOllama(model="llama3.2", temperature=0.9)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


loader = CSVLoader(file_path='./resources/OutdoorClothingCatalog_1000.csv')
document = loader.load()
db = DocArrayInMemorySearch.from_documents(
    document,
    embedding_model
)

query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)
print(f"docs = {docs}")
print("---------------------------")

retriever = db.as_retriever()
qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.")


print(f"response = {response}")
print("---------------------------")

# The retriever retrieves relevant documents from a knowledge base or vector store.
# The chain type defines how the documents retrieved by the retriever are cprocessed
# before generating the answer. Here all are "stuffed" into LLM context. Other options
# are MapReduce, Refine, Map_Rerank
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",

    verbose=True
)
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
response = qa_stuff.run(query)
print(f"response = {response}")
print("---------------------------")

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embedding_model,
).from_loaders([loader])

query = "Please list all your shirts with sun protection in a table in \
markdown and summarize each one."

response = index.query(query, llm=llm)
print(f"response = {response}")


# The three responses are similar in that they all answer the same query using
# the documents, but they differ in how the query is processed and the documents
# are retrieved or used.
# When to Use Each:
# 1.  First Response : Use when you need complete control over how documents are
#                      retrieved and included in the LLM’s prompt.
# 2.  Second Response: Use when you want an automated way to retrieve, process,
#                      and query documents while still having access to the chain’s modular components.
# 3.  Third Response : Use when simplicity and ease of use are your priority,
#                      and you’re comfortable with the VectorstoreIndexCreator handling everything.

