# Notes

Agents use `ReAct = Reason + Act`, to answer the questions.

The LLM understands the user query, reasons and acts on it. It can form another query and use another LLM to get answer to it. This collectively is the Agent.
Thus between the user query and the agent response, there could be multiple calls that happens to different llms/tools.

* langgraph_agent_01 and langgraph_agent_02 use: https://langchain-ai.github.io/langgraph/tutorials/introduction/#setup
* langgraph_agent_03 use: https://learn.deeplearning.ai/courses/ai-agents-in-langgraph

