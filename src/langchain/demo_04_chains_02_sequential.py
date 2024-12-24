import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains.sequential import SequentialChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough



llm = ChatOllama(model="llama3.2", temperature=0.9)
output_parser = StrOutputParser()

prompt_01 = ChatPromptTemplate.from_template(
    """Translate the following review to english. Provide only the translation without any other text:
    {review}."""
)
chain_01 = prompt_01 | llm | output_parser

prompt_02 = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence: \n{english_review}"
)
chain_02 = prompt_02 | llm | output_parser

prompt_03 = ChatPromptTemplate.from_template(
    "What language is the following review: \n{review}. Provide a one word answer"
)
chain_03 = prompt_03 | llm | output_parser

prompt_04 = ChatPromptTemplate.from_template(
    """Write a 5 word follow up message response to the following summary in the {language}:
    Summary: {summary}
    """
)
chain_04 = prompt_04 | llm | output_parser

review = """Me gustaría decir que soy un principiante en los placeres del levantamiento de pesas. \
Esta configuración funciona bien en muchos niveles. El bastidor es muy fácil de configurar. \
sin un montón de piezas que juntar. Lo tuve listo en menos de 2 horas como máximo."""

runnable_seq = (
    RunnablePassthrough()
    | {
        "review": itemgetter("review"),  # preserve original review
        "english_review": chain_01,
    }
    | {
        "review": itemgetter("review"),  # keep original review
        "english_review": itemgetter("english_review"),
        "summary": chain_02,
        "language": chain_03,
    }
    | {
        "review": itemgetter("review"),  # keep original review in the final response
        "english_review": itemgetter("english_review"), # keep english_review in the final response
        "summary": itemgetter("summary"),  # keep original review
        "language": itemgetter("language"),
        "follow_up": chain_04,
    }
)

response = runnable_seq.invoke({ "review": review })
print(json.dumps(response))
print("---")
print(f"Original Review = {response['review']}")
print("---")
print(f"Language = {response['language']}")
print("---")
print(f"English Review = {response['english_review']}")
print("---")
print(f"Summary = {response['summary']}")
print("---")
print(f"Follow Up = {response['follow_up']}")
