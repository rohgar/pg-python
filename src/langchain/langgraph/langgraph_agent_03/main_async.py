import warnings
# duckduckgo-search latest package warning: UserWarning: 'api' backend is deprecated, using backend='auto'
warnings.filterwarnings('ignore', message="'api' backend is deprecated")

# ---------------------------------------------------------------------------------
# In main.py, we streamed events in the result i.e. all intermediate step results.
# We can also stream tokens. For this we need an async checkpointer.
# The token events are of type `on_chat_model_stream`
# Currently this code does not work as ChatOllama does not support token streaming
# Known issues: https://github.com/langchain-ai/langchain/issues/26971
# ---------------------------------------------------------------------------------

from agent import Agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import asyncio

import sys
sys.path.append("..")
from my_tools import get_tools


prompt = """You are a smart research assistant. Use the available tools to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
Give concise answers.
"""

tools = get_tools()
model = ChatOllama(model="llama3.2", temperature=0.2, streaming=True)

# session - this will allow us to have multiple conversations with different
# user simultaneously
thread = {"configurable": {"thread_id": "2"}}

async def use_token_stream():
    # memory = await AsyncSqliteSaver.from_conn_string(":memory:")
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        agent = Agent(model, tools, system_msg_prompt=prompt, checkpointer=memory)

        query = "Whats the weather in SF?"
        messages = [HumanMessage(content=query)]
        async for event in agent.graph.astream_events({"messages": messages}, thread, version="v1"):
            kind = event["event"]
            print(f"kind = {kind}")
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # Empty content in the context of OpenAI means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content
                    print(content, end="|")

if __name__ == "__main__":
    asyncio.run(use_token_stream())
