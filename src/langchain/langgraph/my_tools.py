from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.agents import Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
import os
from datetime import date


wikipedia_tool = Tool(
    name="wikipedia",
    func=WikipediaAPIWrapper().run,
    description="Useful for when you need to look up the songwriters, genre, \
                and producers for a song on wikipedia",
)

duckduckgo_tool = Tool(
    name="DuckDuckGo_Search",
    func=DuckDuckGoSearchRun().run,
    description="Useful for when you need to do a search on the internet to find \
                information that the other tools can't find.",
)


# has to have the input string as an empty string will be passed by the agent if
# there is no input
def get_todays_date(input: str) -> str:
    return str(date.today())
my_datetime_tool = Tool(
    name="my_datetime",
    func=get_todays_date,
    description="Returns todays date, use this for any questions related to \
    knowing todays date. The input should always be an empty string, and this \
    function will always return todays date - any date mathematics should occur \
    outside this function.",
)

python_repl_tool = PythonREPLTool()

def get_tools(include_tavily=False):
    if include_tavily:
        if not os.getenv('TAVILY_API_KEY'):
            raise Exception("'TAVILY_API_KEY' is not set")
        return [my_datetime_tool, wikipedia_tool, TavilySearchResults(max_results=1), python_repl_tool]
    return [my_datetime_tool, wikipedia_tool, duckduckgo_tool, python_repl_tool]