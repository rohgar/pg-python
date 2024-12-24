from langchain_ollama import ChatOllama
from langchain_core.messages import trim_messages
from langgraph.prebuilt import create_react_agent
from my_tools import MyTools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class MyChain:

    def __init__(self, my_tools: MyTools):

        self.model = ChatOllama(model="llama3.2", temperature=0.2)
        memory_trimmer = trim_messages(
            max_tokens=500,
            strategy="last",
            token_counter=self.model,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

        self.model.bind_tools(my_tools.tools)
        self.chain = memory_trimmer | self.model


