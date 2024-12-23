from langchain_community.tools.tavily_search import TavilySearchResults
import os


class MyTools:

    def __init__(self):
        # create an agent that uses Tavily api to search the internet
        # for the user's query.
        if not os.getenv('TAVILY_API_KEY'):
            raise Exception("'TAVILY_API_KEY' is not set")

        search = TavilySearchResults(max_results=2)
        self.tools = [search]
