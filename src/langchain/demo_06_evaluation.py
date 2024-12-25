import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain_ollama import ChatOllama
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_huggingface import HuggingFaceEmbeddings
from IPython.display import display, Markdown
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain

# This script does the same as demo_05_q_and_a.py, but shows different ways of doing the same thing.


llm = ChatOllama(model="llama3.2", temperature=0.9)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


loader = CSVLoader(file_path='./resources/OutdoorClothingCatalog_1000.csv')
data = loader.load()

vectorstore = DocArrayInMemorySearch.from_documents(
    data,
    embedding_model
)
retriever = vectorstore.as_retriever()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embedding_model,
).from_loaders([loader])

# The retriever retrieves relevant documents from a knowledge base or vector store.
# The chain type defines how the documents retrieved by the retriever are cprocessed
# before generating the answer. Here all are "stuffed" into LLM context. Other options
# are MapReduce, Refine, Map_Rerank
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    verbose=True,
    chain_type_kwargs={
        "document_separator": "<<<<<>>>>>"
    }
)


# --------------------------------------
# Evaluation
# --------------------------------------

# Manual sample creation:
# We take some data points, eg. here we use data[10] and data[11] to create some samples:

examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

# Automated sample creation:
# We do this using `QAGenerateChain`

example_gen_chain = QAGenerateChain.from_llm(llm)


def create_doc_dicts(data, n=5):
    output = []
    for item in data[:n]:
        output.append({"doc": item})
    return output

auto_gen_examples = example_gen_chain.apply_and_parse(
    create_doc_dicts(data)
)

auto_gen_examples_2 = []
for item in auto_gen_examples:
    auto_gen_examples_2.append(item['qa_pairs'])

# gather the examples together
examples += auto_gen_examples_2

for item in examples:
    print(item)

# --------------------------------------
# Manual Evaluation
# --------------------------------------

print(retrieval_qa.invoke(examples[0]))

# --------------------------------------
# LLM assisted Evaluation
# --------------------------------------

predictions = retrieval_qa.batch(examples)
eval_chain = QAEvalChain.from_llm(llm)

graded_outputs = eval_chain.evaluate(examples, predictions)
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['results'])
    print("---")

# This data can be used as a dataset.
